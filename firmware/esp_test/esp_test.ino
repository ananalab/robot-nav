// Test du module Wi-Fi ESP8266 : essaie plusieurs vitesses et affiche ce que le module renvoie.
// Branchement : TX du module -> D3, D11 -> pont diviseur -> RX du module, 5V et GND communs.
// Après le test : remettre le vrai firmware avec scripts/flash_firmware.
#include <SoftwareSerial.h>

SoftwareSerial esp(3, 11);  // RX = D3, TX = D11
const long BAUDS[] = {115200, 74880, 57600, 38400, 19200, 9600};

void listen(unsigned long ms) {
  int count = 0, printable = 0;
  char text[80];
  int n = 0;
  const unsigned long start = millis();
  while (millis() - start < ms) {
    while (esp.available() > 0) {
      const char c = static_cast<char>(esp.read());
      ++count;
      if (c >= 32 && c < 127) ++printable;
      if (n < 79 && c >= 32 && c < 127) text[n++] = c;
    }
  }
  text[n] = '\0';
  Serial.print(F("  octets recus: "));
  Serial.print(count);
  Serial.print(F(" (lisibles: "));
  Serial.print(printable);
  Serial.print(F(") texte: "));
  Serial.println(text);
}

void setup() {
  const int motor_pins[4] = {5, 6, 9, 10};
  for (int pin : motor_pins) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);  // moteurs arrêtés
  }
  Serial.begin(115200);
  Serial.println(F("=== test ESP8266 ==="));
  for (long baud : BAUDS) {
    esp.begin(baud);
    Serial.print(F("vitesse "));
    Serial.print(baud);
    Serial.println(F(" : envoi AT"));
    esp.print("AT\r\n");
    listen(700);
  }
  Serial.println(F("redemarrage du module (AT+RST a 115200), ecoute a 74880 et 115200 :"));
  esp.begin(115200);
  esp.print("AT+RST\r\n");
  esp.begin(74880);
  listen(1500);
  esp.begin(115200);
  listen(1500);
  // Écoute seule sur D11 (TX du SoftwareSerial mis sur D12, non branché) : si le module
  // parle de ce côté, ses fils TX et RX sont inversés.
  SoftwareSerial probe(11, 12);
  const long PROBE_BAUDS[] = {115200, 74880, 9600};
  for (long baud : PROBE_BAUDS) {
    probe.begin(baud);
    probe.listen();
    Serial.print(F("ecoute sur D11 a "));
    Serial.print(baud);
    Serial.print(F(" : "));
    int count = 0;
    const unsigned long start = millis();
    while (millis() - start < 1500) {
      while (probe.available() > 0) {
        probe.read();
        ++count;
      }
    }
    Serial.print(count);
    Serial.println(F(" octets"));
  }
  esp.listen();
  Serial.print(F("niveau de D3 au repos (1 = normal, le TX du module est haut au repos) : "));
  pinMode(3, INPUT);
  Serial.println(digitalRead(3));
  pinMode(11, INPUT);
  Serial.print(F("niveau de D11 en entree : "));
  Serial.println(digitalRead(11));
  Serial.println(F("=== fin ==="));
}

void loop() {}
