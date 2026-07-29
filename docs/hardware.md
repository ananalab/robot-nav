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
|---|---|
| VCC | 5V |
| GND | GND |
| TX | D3 (SoftwareSerial RX) |
| RX | D11 through a 1 kΩ / 2 kΩ divider |
| EN | 3.3V |

Firmware settings (`firmware/robot_fw/config.h`): both motors inverted, 300 ms watchdog, PWM ramp of 5 per 5 ms, ESP8266 at 19200 baud, radio power 20.

### Identified motor model

`robot_tool calib --real`, camera-based step responses.

| | Left | Right |
|---|---|---|
| Gain (m/s per unit command) | 0.584 | 0.451 |
| Dead zone (rolling wheel) | 0.1 | 0.1 |
| Time constant / delay | 0.126 s / 0.113 s | |

Static friction is handled in the wheel mapper: a turning wheel receives at least 0.28, and 0.38 during the first 4 cycles after a start.

## Arena

| | |
|---|---|
| Drivable area | 1.47 × 0.99 m (`arena.bounds`) |
| Floor markers | ArUco DICT_4X4_50, ids 0–3, 114 mm, one per corner |
| Marker centres | (0, 0), (1.335, 0), (1.332, 0.850), (−0.009, 0.850) m |
| Obstacles | flat red sheets, 10 × 10 cm, detected in HSV |
| Obstacle inflation | robot radius + 1 cm |

## Camera

| | |
|---|---|
| Sensor | iPhone, ultra-wide lens through Continuity Camera (Center Stage on) |
| Stream | 1920 × 1080, 30 fps |
| Height | 1.41 m above the floor, above the arena centre |
| Intrinsics | fx ≈ 911 px, ChArUco calibration, RMS 0.19 px |
| Floor pose | 4 floor markers, reprojection error 2.6 px |
| Latency | 0.15 s (assumed) |
