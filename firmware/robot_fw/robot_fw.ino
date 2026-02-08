// Firmware de la voiture : applique les PWM reçues de l'ordinateur, rien d'autre.
// Les commandes arrivent par le câble USB ou par le Wi-Fi (module ESP8266, en UDP).
//
// - boucle non bloquante, pas de classe String, buffers fixes de 32 octets ;
// - trames invalides ignorées ;
// - watchdog 300 ms : sans commande valide, moteurs arrêtés (et trame "W" sur l'USB) ;
// - LED 13 allumée tant que les commandes arrivent (si USE_LED vaut 1).
// Ne JAMAIS inclure Servo.h : cette bibliothèque désactive le PWM sur D9 et D10.

#include "config.h"
#include "fw_protocol.h"

#if USE_WIFI
#include <SoftwareSerial.h>
#include "wifi_secrets.h"  // noms et mots de passe Wi-Fi (copier wifi_secrets.example.h)
static SoftwareSerial esp(ESP_RX_PIN, ESP_TX_PIN);
#endif

#define STR_(x) #x
#define STR(x) STR_(x)

// Une ligne en cours de réception (une par source : USB et Wi-Fi).
struct LineReader {
  char buffer[FW_LINE_MAX];
  uint8_t length;
  bool overflow;
};
static LineReader usb_line = {{0}, 0, false};
static LineReader wifi_line = {{0}, 0, false};

static unsigned long last_valid_ms = 0;
static bool watchdog_fired = true;  // moteurs arrêtés au démarrage

// Démarrage en douceur : la PWM appliquée rejoint la PWM demandée par petits pas.
// Un démarrage brutal fait chuter la pile 9 V et redémarrer le module Wi-Fi.
static int target_left = 0, target_right = 0;  // PWM demandées
static int out_left = 0, out_right = 0;        // PWM appliquées
static unsigned long last_ramp_ms = 0;

// Envoie "<body>*CK\n" sur l'USB.
static void sendFrame(const char* body) {
  char out[FW_LINE_MAX];
  fwFrame(body, out);
  Serial.print(out);
  Serial.print('\n');
}

// pwm dans [-255, 255] ; positif = marche avant.
static void driveMotor(uint8_t fwd_pin, uint8_t rev_pin, int pwm) {
  if (pwm > 0) {
    digitalWrite(rev_pin, LOW);
    analogWrite(fwd_pin, pwm);
  } else if (pwm < 0) {
    digitalWrite(fwd_pin, LOW);
    analogWrite(rev_pin, -pwm);
  } else {
    // Arrêt : toutes les entrées à l'état bas (le L298N freine).
    digitalWrite(fwd_pin, LOW);
    digitalWrite(rev_pin, LOW);
  }
}

static void applyMotors(int left, int right) {
  if (INVERT_LEFT) left = -left;
  if (INVERT_RIGHT) right = -right;
  if (SWAP_SIDES) {
    const int tmp = left;
    left = right;
    right = tmp;
  }
  driveMotor(LEFT_FWD_PIN, LEFT_REV_PIN, left);
  driveMotor(RIGHT_FWD_PIN, RIGHT_REV_PIN, right);
}

// Arrêt immédiat (pas de rampe).
static void stopMotors() {
  target_left = target_right = out_left = out_right = 0;
  applyMotors(0, 0);
}

static int stepToward(int value, int target) {
  if (value < target) return min(value + RAMP_STEP, target);
  if (value > target) return max(value - RAMP_STEP, target);
  return value;
}

// Appelée à chaque tour de loop() : avance la PWM appliquée d'un pas toutes les RAMP_MS.
static void updateRamp() {
  if (millis() - last_ramp_ms < RAMP_MS) return;
  last_ramp_ms = millis();
  if (out_left == target_left && out_right == target_right) return;
  out_left = stepToward(out_left, target_left);
  out_right = stepToward(out_right, target_right);
  applyMotors(out_left, out_right);
}

// Les acquittements ne partent que sur l'USB (en Wi-Fi, la liaison va dans un seul sens).
static void sendAck(int seq, bool from_usb) {
  if (!SEND_ACK || !from_usb) return;
  char body[12];
  snprintf(body, sizeof(body), "A,%d", seq);
  sendFrame(body);
}

static void handleLine(const char* line, bool from_usb) {
  FwCommand cmd;
  if (!fwParseLine(line, cmd)) return;  // trame invalide : ignorée
  last_valid_ms = millis();
  watchdog_fired = false;
  if (cmd.type == FW_MOTOR) {
    target_left = cmd.left;
    target_right = cmd.right;
    sendAck(cmd.seq, from_usb);
  } else if (cmd.type == FW_STOP) {
    stopMotors();
  } else if (cmd.type == FW_PING) {
    sendAck(cmd.seq, from_usb);
  }
}

// Ajoute un caractère à la ligne ; renvoie true quand une ligne complète et valide est prête.
static bool addChar(LineReader& r, char c) {
  if (c == '\r') return false;
  if (c == '\n') {
    const bool ok = !r.overflow;
    r.buffer[r.length] = '\0';
    r.length = 0;
    r.overflow = false;
    return ok;
  }
  if (r.length < FW_LINE_MAX - 1) {
    r.buffer[r.length++] = c;
  } else {
    r.overflow = true;  // ligne trop longue : on l'ignorera entièrement
  }
  return false;
}

static void readUsb() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (addChar(usb_line, c)) handleLine(usb_line.buffer, true);
  }
}

#if USE_WIFI
// Le module annonce chaque paquet UDP reçu par "+IPD,<longueur>:<données>".
static uint8_t ipd_match = 0;     // nombre de caractères de "+IPD," déjà reconnus
static bool reading_length = false;
static int packet_length = 0;
static int packet_remaining = 0;  // octets de données restant à lire dans le paquet

static void readWifi() {
  const char* header = "+IPD,";
  while (esp.available() > 0) {
    const char c = static_cast<char>(esp.read());
    if (packet_remaining > 0) {
      --packet_remaining;
      if (addChar(wifi_line, c)) handleLine(wifi_line.buffer, false);
    } else if (reading_length) {
      if (c >= '0' && c <= '9') {
        packet_length = packet_length * 10 + (c - '0');
      } else {
        if (c == ':') packet_remaining = packet_length;
        reading_length = false;
      }
    } else if (c == header[ipd_match]) {
      if (++ipd_match == 5) {
        ipd_match = 0;
        reading_length = true;
        packet_length = 0;
      }
    } else {
      ipd_match = (c == '+') ? 1 : 0;
    }
  }
}

// Attend que le module écrive "expected" (ou abandonne après timeout_ms).
static bool espWait(const char* expected, unsigned long timeout_ms) {
  const unsigned long start = millis();
  const size_t n = strlen(expected);
  size_t matched = 0;
  while (millis() - start < timeout_ms) {
    while (esp.available() > 0) {
      const char c = static_cast<char>(esp.read());
      if (c == expected[matched]) {
        if (++matched == n) return true;
      } else {
        matched = (c == expected[0]) ? 1 : 0;
      }
    }
  }
  return false;
}

static bool espCommand(const char* command, const char* expected, unsigned long timeout_ms) {
  esp.print(command);
  esp.print("\r\n");
  return espWait(expected, timeout_ms);
}

// Configure le module (point d'accès ou connexion à la box), écoute UDP, affiche l'IP sur l'USB.
static void setupWifi() {
  // Redémarrage du module (il redémarre à 115200 ou 9600 bauds selon sa version), puis
  // passage à ESP_BAUD : SoftwareSerial ne sait pas recevoir de façon fiable à 115200.
  const long bauds[3] = {ESP_BAUD, 115200, 9600};
  for (int i = 0; i < 3; ++i) {
    esp.begin(bauds[i]);
    esp.print("AT+RST\r\n");
  }
  delay(2500);
  for (int i = 1; i < 3; ++i) {
    esp.begin(bauds[i]);
    esp.print("AT+UART_CUR=" STR(ESP_BAUD) ",8,1,0,0\r\n");
    delay(300);
  }
  esp.begin(ESP_BAUD);

  Serial.println(espCommand("AT", "OK", 2000) ? F("wifi: module OK") : F("wifi: le module ne repond pas"));
#if WIFI_ACCESS_POINT
  // Le module crée son propre réseau : l'ordinateur s'y connecte (adresse du robot 192.168.4.1).
  espCommand("AT+CWMODE=2", "OK", 2000);
  esp.print("AT+CWSAP=\"" AP_SSID "\",\"" AP_PASSWORD "\",5,3,1\r\n");  // 1 seul appareil connecté
  Serial.println(espWait("OK", 5000) ? F("wifi: reseau " AP_SSID " cree") : F("wifi: creation du reseau impossible"));
#else
  // Le module rejoint la box.
  espCommand("AT+CWMODE=1", "OK", 2000);
  esp.print("AT+CWJAP=\"" WIFI_SSID "\",\"" WIFI_PASSWORD "\"\r\n");
  Serial.println(espWait("OK", 20000) ? F("wifi: connecte") : F("wifi: connexion impossible"));
#endif
  espCommand("AT+SLEEP=0", "OK", 1000);  // pas de mise en veille : latence plus faible
  espCommand("AT+RFPOWER=" STR(ESP_RF_POWER), "OK", 1000);  // émission moins forte : moins de pics de courant
  espCommand("AT+CIPMUX=0", "OK", 1000);
  espCommand("AT+CIPSTART=\"UDP\",\"0.0.0.0\"," STR(UDP_PORT) "," STR(UDP_PORT) ",2", "OK", 3000);

  // Réponse attendue : +CIFSR:APIP,"192.168.4.1" (point d'accès) ou +CIFSR:STAIP,"..." (box)
  esp.print("AT+CIFSR\r\n");
  if (espWait(WIFI_ACCESS_POINT ? "APIP,\"" : "STAIP,\"", 3000)) {
    Serial.print(F("wifi: IP du robot = "));
    const unsigned long start = millis();
    while (millis() - start < 500) {
      if (esp.available() > 0) {
        const char c = static_cast<char>(esp.read());
        if (c == '"') break;
        Serial.print(c);
      }
    }
    Serial.println();
  }
}
#endif

void setup() {
  pinMode(LEFT_FWD_PIN, OUTPUT);
  pinMode(LEFT_REV_PIN, OUTPUT);
  pinMode(RIGHT_FWD_PIN, OUTPUT);
  pinMode(RIGHT_REV_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  stopMotors();
  Serial.begin(SERIAL_BAUD);
#if USE_WIFI
  setupWifi();
#endif
  // Un clignotement, puis "prêt".
  digitalWrite(LED_PIN, USE_LED ? HIGH : LOW);
  delay(100);
  digitalWrite(LED_PIN, LOW);
  sendFrame("B");
}

void loop() {
  readUsb();
#if USE_WIFI
  readWifi();
#endif
  const bool alive = millis() - last_valid_ms <= WATCHDOG_MS;
  updateRamp();
  if (!alive && !watchdog_fired) {
    stopMotors();
    sendFrame("W");
    watchdog_fired = true;
  }
  digitalWrite(LED_PIN, (USE_LED && alive) ? HIGH : LOW);
}
