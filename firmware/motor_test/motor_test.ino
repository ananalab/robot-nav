// Test des moteurs sans ordinateur : au démarrage, la carte fait une seule fois
//   1) les deux roues en avant (environ 10 cm tout droit) ;
//   2) chaque sortie du L298N seule, dans chaque sens.
// Pour recommencer : bouton RESET de la carte. Les étapes s'affichent sur l'USB (115200 bauds).
// Broches : celles du câblage de docs/hardware.md.

const int A_FWD = 6;   // L298N IN1 (sortie OUT1/OUT2, prévue pour la roue droite)
const int A_REV = 5;   // L298N IN2
const int B_FWD = 9;   // L298N IN4 (sortie OUT3/OUT4, prévue pour la roue gauche)
const int B_REV = 10;  // L298N IN3
const int PWM = 200;

void allOff() {
  analogWrite(A_FWD, 0);
  analogWrite(A_REV, 0);
  analogWrite(B_FWD, 0);
  analogWrite(B_REV, 0);
}

// Met les broches données à PWM pendant ms millisecondes, puis tout arrête et attend 2 s.
void step(const char* name, int pin1, int pin2, int ms) {
  Serial.println(name);
  allOff();
  if (pin1 >= 0) analogWrite(pin1, PWM);
  if (pin2 >= 0) analogWrite(pin2, PWM);
  delay(ms);
  allOff();
  delay(2000);
}

void setup() {
  pinMode(A_FWD, OUTPUT);
  pinMode(A_REV, OUTPUT);
  pinMode(B_FWD, OUTPUT);
  pinMode(B_REV, OUTPUT);
  allOff();
  Serial.begin(115200);
  Serial.println("test moteurs dans 5 s");
  delay(5000);
  step("1) les deux roues en avant", A_FWD, B_FWD, 400);
  step("2) sortie A (D6/D5) en avant", A_FWD, -1, 700);
  step("3) sortie A (D6/D5) en arriere", A_REV, -1, 700);
  step("4) sortie B (D9/D10) en avant", B_FWD, -1, 700);
  step("5) sortie B (D9/D10) en arriere", B_REV, -1, 700);
  Serial.println("fin");
}

void loop() {}
