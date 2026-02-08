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
