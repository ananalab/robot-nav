// Réglages du firmware. Câblage : voir docs/hardware.md (driver L298N).
// NE PAS changer les broches sans vérifier le câblage réel.
#ifndef ROBOT_FW_CONFIG_H
#define ROBOT_FW_CONFIG_H

#define SERIAL_BAUD 115200

// Moteur DROIT = sortie OUT1/OUT2 du L298N.
//   marche avant = PWM sur RIGHT_FWD_PIN, RIGHT_REV_PIN à 0 (tutoriel SunFounder : D6 haut, D5 bas)
#define RIGHT_FWD_PIN 6   // L298N IN1
#define RIGHT_REV_PIN 5   // L298N IN2
// Moteur GAUCHE = sortie OUT3/OUT4 du L298N.
//   marche avant = PWM sur LEFT_FWD_PIN, LEFT_REV_PIN à 0 (tutoriel SunFounder : D9 haut, D10 bas)
#define LEFT_FWD_PIN 9    // L298N IN4
#define LEFT_REV_PIN 10   // L298N IN3

// À ajuster pendant le test de téléop (roues en l'air) :
#define INVERT_LEFT 1     // 1 si la roue gauche tourne à l'envers
#define INVERT_RIGHT 1    // 1 si la roue droite tourne à l'envers
#define SWAP_SIDES 0      // 1 si la commande "gauche" fait tourner la roue droite

#define WATCHDOG_MS 300   // sans commande valide pendant ce temps : arrêt + trame W
#define RAMP_STEP 5       // démarrage en douceur : la PWM change d'au plus 5 ...
#define RAMP_MS 5         // ... toutes les 5 ms (0 -> 255 en 260 ms)
#define SEND_ACK 1        // 1 = acquitter M et P (mesure de latence aller-retour)
#define LED_PIN 13        // allumée tant que les commandes arrivent ...
#define USE_LED 0         // ... si USE_LED vaut 1 (0 = éteinte, pour économiser la pile)

// Wi-Fi : module ESP8266 SunFounder (TX du module -> D3, RX du module <- D11 par pont diviseur).
// Mettre USE_WIFI à 0 pour n'utiliser que le câble USB.
#define USE_WIFI 1
#define WIFI_ACCESS_POINT 1  // 1 = le robot crée son réseau (AP_SSID) ; 0 = il rejoint la box (WIFI_SSID)
#define ESP_RX_PIN 3      // reçoit ce qu'envoie le module (son TX)
#define ESP_TX_PIN 11     // envoie vers le module (son RX), via le pont 1 kOhm / 2 kOhm
#define ESP_BAUD 19200    // débit carte <-> module (SoftwareSerial n'est pas fiable à 115200)
#define UDP_PORT 4210     // port UDP où le robot reçoit les commandes
#define ESP_RF_POWER 20   // puissance d'émission du module (0 à 82) : basse pour économiser la pile, assez pour 2 m

// Capteurs en bonus (non utilisés par ce firmware), voir docs/hardware.md :
//   ultrason Trig D2, Echo D4 ; infrarouges D7 (gauche), D8 (droite) ; suivi de ligne A0.

#endif  // ROBOT_FW_CONFIG_H
