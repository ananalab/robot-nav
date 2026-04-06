#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core.hpp"

// Protocole série, côté PC. Le côté firmware est dans firmware/robot_fw/fw_protocol.h.

// XOR de tous les octets.
std::uint8_t checksum(std::string_view body);
// Ajoute "*CK" au corps de la trame.
std::string frame(std::string_view body);

std::string encodeMotor(int left_pwm, int right_pwm, int seq);
std::string encodeStop();
std::string encodePing(int seq);

// Commande normalisée [-1, 1] -> PWM entière [-255, 255] (arrondie, saturée).
int toPwm(double u);

// Trames envoyées par le robot.
struct RobotMessage {
  enum class Type { Ack, Watchdog, Boot };
  Type type = Type::Ack;
  int seq = 0;
};

// Analyse une ligne reçue (sans '\n') ; vide si invalide (checksum, format...).
std::optional<RobotMessage> parseRobotLine(std::string_view line);

// Liaison série USB (8N1, non bloquante). POSIX (termios), macOS et Linux.
// Attention : ouvrir le port redémarre la carte UNO (signal DTR). Attendre la trame "B"
// (ou 2 s) avant d'envoyer des commandes : ProtocolRobot::waitReady() le fait.

class SerialTransport : public ITransport {
 public:
  // Ouvre le port (ex. "/dev/cu.usbserial-110", "/dev/ttyUSB0", "COM3"). Lève en cas d'échec.
  SerialTransport(const std::string& port, int baud);
  ~SerialTransport() override;
  SerialTransport(const SerialTransport&) = delete;
  SerialTransport& operator=(const SerialTransport&) = delete;

  bool writeLine(std::string_view line) override;
  std::optional<std::string> readLine() override;

 private:
  void readAvailable();

  int fd_ = -1;
  std::string pending_;
};

// Ports série plausibles pour la carte (usbserial, usbmodem, ttyUSB, ttyACM).
std::vector<std::string> listSerialPorts();

// Liaison Wi-Fi : chaque ligne part dans un paquet UDP vers le module ESP8266 du robot.
// La liaison va dans un seul sens (le robot ne répond pas) : pas d'acquittement ni de "B".
class UdpTransport : public ITransport {
 public:
  UdpTransport(const std::string& robot_ip, int port);
  ~UdpTransport() override;
  UdpTransport(const UdpTransport&) = delete;
  UdpTransport& operator=(const UdpTransport&) = delete;

  bool writeLine(std::string_view line) override;
  std::optional<std::string> readLine() override { return std::nullopt; }

 private:
  int socket_ = -1;
  std::string ip_;
  int port_;
};

// ProtocolRobot : le robot vu à travers le protocole série.
// C'est LA MÊME classe en simulation (sur SimFirmware) et en réel (sur SerialTransport) :
// la téléop et la calibration des moteurs testées en sim marcheront donc sur le vrai robot.

struct ProtocolRobotConfig {
  double resend_rate = 25.0;   // renvoi de la dernière commande (Hz), pour nourrir le watchdog
  double ready_timeout = 2.0;  // attente max de la trame "B" après ouverture du port (s)
};

ProtocolRobotConfig loadProtocolRobotConfig(const ConfigNode& section);

class ProtocolRobot {
 public:
  ProtocolRobot(ITransport& transport, IClock& clock, const ProtocolRobotConfig& config);
  // Envoie "S" : le robot s'arrête quand l'objet est détruit (RAII).
  ~ProtocolRobot();
  ProtocolRobot(const ProtocolRobot&) = delete;
  ProtocolRobot& operator=(const ProtocolRobot&) = delete;

  // Attend la trame "B" (la carte redémarre à l'ouverture du port) ou ready_timeout.
  bool waitReady();
  void send(const WheelCmd& cmd);
  void stop();
  // À appeler à chaque cycle : lit les réponses, renvoie la commande si besoin.
  void poll();

  double lastRoundTrip() const { return last_rtt_; }  // s, -1 si inconnu
  int watchdogCount() const { return watchdog_count_; }

 private:
  void transmit();

  ITransport& transport_;
  IClock& clock_;
  ProtocolRobotConfig config_;
  WheelCmd last_;
  int seq_ = 0;
  Time last_sent_ = -1e9;
  std::array<Time, 256> sent_at_{};
  double last_rtt_ = -1.0;
  int watchdog_count_ = 0;
  bool boot_seen_ = false;
};
