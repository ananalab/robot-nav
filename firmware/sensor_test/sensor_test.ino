// Programme de TEST des capteurs (ne fait jamais tourner les moteurs).
// Affiche 5 fois par seconde : infrarouge gauche / droit, suivi de ligne, distance ultrason.
// Voir docs/hardware.md. Après le test, remettre le vrai firmware : scripts/flash_firmware
//
// Broches (docs/hardware.md) : ultrason Trig D2, Echo D4 ; infrarouges D7 (gauche), D8 (droite) ;
// suivi de ligne A0. Moteurs : D5, D6, D9, D10 maintenus à l'arrêt.

const int TRIG_PIN = 2;
const int ECHO_PIN = 4;
const int IR_LEFT_PIN = 7;
const int IR_RIGHT_PIN = 8;
const int LINE_PIN = A0;
const int MOTOR_PINS[] = {5, 6, 9, 10};

void setup() {
  for (int pin : MOTOR_PINS) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);  // moteurs arrêtés
  }
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(IR_LEFT_PIN, INPUT);
  pinMode(IR_RIGHT_PIN, INPUT);
  pinMode(LINE_PIN, INPUT);
  Serial.begin(115200);
  Serial.println(F("Test des capteurs : approchez la main, posez la voiture sur du noir / du blanc."));
}

// Distance ultrason en cm (0 = rien reçu : fil Echo ou Trig mal branché, ou rien devant).
long ultrasonicCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  const unsigned long echo_us = pulseIn(ECHO_PIN, HIGH, 30000UL);  // 30 ms max (~5 m)
  return static_cast<long>(echo_us / 58UL);                        // aller-retour du son
}

void loop() {
  // Les modules infrarouges du kit mettent leur sortie à 0 (LOW) quand ils voient un obstacle.
  Serial.print(F("IR gauche: "));
  Serial.print(digitalRead(IR_LEFT_PIN) == LOW ? F("OBSTACLE") : F("libre   "));
  Serial.print(F(" | IR droite: "));
  Serial.print(digitalRead(IR_RIGHT_PIN) == LOW ? F("OBSTACLE") : F("libre   "));
  Serial.print(F(" | ligne: "));
  Serial.print(digitalRead(LINE_PIN) == HIGH ? F("1") : F("0"));
  Serial.print(F(" | ultrason: "));
  Serial.print(ultrasonicCm());
  Serial.println(F(" cm"));
  delay(200);
}
