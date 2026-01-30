# Hardware

## Robot

| | |
|---|---|
| Base | SunFounder 3-in-1 Arduino car, differential drive |
| Controller | Arduino Uno R3 |
| Motor driver | L298N |
| Radio | ESP8266 (AT firmware), access point `robot_nav`, UDP port 4210 |
| Footprint | ≈ 20 × 20 cm, bounding radius 0.12 m (from the axle centre) |
| Wheel base | 0.140 m |
| Wheel diameter | 0.063 m |
| Roof marker | ArUco DICT_4X4_50, id 10, 85 mm, 0.1375 m above the floor, 7.5 mm behind the axle |
| Unused sensors | HC-SR04, 2 IR obstacle sensors, line tracker (unpowered) |

### Power

| Source | Supplies |
|---|---|
| 9 V alkaline battery → barrel jack → VIN → L298N +12V | motors |
| USB power bank → Arduino USB | Arduino and ESP8266 |

A single 9 V battery cannot supply the motors and the ESP8266 at the same time: the radio drops out when the motors start.

### Wiring

| L298N | Arduino / other |
|---|---|
| OUT1, OUT2 | right motor |
| OUT3, OUT4 | left motor |
| IN1 | D6 (PWM) |
| IN2 | D5 (PWM) |
| IN3 | D10 (PWM) |
| IN4 | D9 (PWM) |
| +12V | VIN |
| GND | GND |
| +5V | not connected |
| ENA, ENB, 5V-EN | jumpers fitted |

| ESP8266 | Arduino |
