#include "hw.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

std::uint8_t checksum(std::string_view body) {
  std::uint8_t ck = 0;
  for (char c : body) ck ^= static_cast<std::uint8_t>(c);
  return ck;
}

std::string frame(std::string_view body) {
  char hex[4];
  std::snprintf(hex, sizeof(hex), "%02X", checksum(body));
  return std::string(body) + "*" + hex;
}

std::string encodeMotor(int left_pwm, int right_pwm, int seq) {
  return frame("M," + std::to_string(std::clamp(left_pwm, -255, 255)) + "," +
               std::to_string(std::clamp(right_pwm, -255, 255)) + "," +
               std::to_string(seq & 0xFF));
}

std::string encodeStop() { return frame("S"); }

std::string encodePing(int seq) { return frame("P," + std::to_string(seq & 0xFF)); }

int toPwm(double u) { return static_cast<int>(std::lround(std::clamp(u, -1.0, 1.0) * 255.0)); }

std::optional<RobotMessage> parseRobotLine(std::string_view line) {
  const std::size_t star = line.find('*');
  if (star == std::string_view::npos || star == 0 || line.size() != star + 3) return std::nullopt;
  const std::string_view body = line.substr(0, star);
  const std::string_view ck = line.substr(star + 1);
  char expected[4];
  std::snprintf(expected, sizeof(expected), "%02X", checksum(body));
  if (ck != std::string_view(expected, 2)) return std::nullopt;

  RobotMessage msg;
  if (body == "W") {
    msg.type = RobotMessage::Type::Watchdog;
    return msg;
  }
  if (body == "B") {
    msg.type = RobotMessage::Type::Boot;
    return msg;
  }
  if (body.size() >= 3 && body.substr(0, 2) == "A,") {
    int seq = 0;
    for (char c : body.substr(2)) {
      if (c < '0' || c > '9') return std::nullopt;
      seq = seq * 10 + (c - '0');
      if (seq > 255) return std::nullopt;
    }
    msg.type = RobotMessage::Type::Ack;
    msg.seq = seq;
    return msg;
  }
  return std::nullopt;
}



namespace {
speed_t toSpeed(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 57600: return B57600;
    case 115200: return B115200;
    default: throw std::runtime_error("débit série non géré : " + std::to_string(baud));
  }
}
}  // namespace

SerialTransport::SerialTransport(const std::string& port, int baud) {
  fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    throw std::runtime_error("impossible d'ouvrir " + port +
                             " (câble branché ? sous Linux : groupe dialout)");
  }
  termios tty{};
  tcgetattr(fd_, &tty);
  cfmakeraw(&tty);  // octets bruts, pas d'écho ni de traitement des fins de ligne
  cfsetispeed(&tty, toSpeed(baud));
  cfsetospeed(&tty, toSpeed(baud));
  tty.c_cflag |= CLOCAL | CREAD;  // 8N1 : 8 bits, pas de parité, 1 bit de stop
  tty.c_cflag &= ~(PARENB | CSTOPB | CSIZE);
  tty.c_cflag |= CS8;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;
  tcsetattr(fd_, TCSANOW, &tty);
  tcflush(fd_, TCIOFLUSH);
}

SerialTransport::~SerialTransport() {
  if (fd_ >= 0) ::close(fd_);
}

bool SerialTransport::writeLine(std::string_view line) {
  const std::string data = std::string(line) + "\n";
  return ::write(fd_, data.data(), data.size()) == static_cast<ssize_t>(data.size());
}

void SerialTransport::readAvailable() {
  char buf[256];
  ssize_t n = 0;
  while ((n = ::read(fd_, buf, sizeof(buf))) > 0) pending_.append(buf, static_cast<std::size_t>(n));
}

std::vector<std::string> listSerialPorts() {
  std::vector<std::string> ports;
  const char* patterns[] = {"cu.usbserial", "cu.usbmodem", "cu.wchusbserial", "ttyUSB", "ttyACM"};
  for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
    const std::string name = entry.path().filename().string();
    for (const char* p : patterns) {
      if (name.rfind(p, 0) == 0) ports.push_back(entry.path().string());
    }
  }
  std::sort(ports.begin(), ports.end());
  return ports;
}


UdpTransport::UdpTransport(const std::string& robot_ip, int port) : ip_(robot_ip), port_(port) {
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_ < 0) throw std::runtime_error("impossible de créer la socket UDP");
}

UdpTransport::~UdpTransport() {
  if (socket_ >= 0) ::close(socket_);
}

bool UdpTransport::writeLine(std::string_view line) {
  sockaddr_in robot{};
  robot.sin_family = AF_INET;
  robot.sin_port = htons(static_cast<uint16_t>(port_));
  if (inet_pton(AF_INET, ip_.c_str(), &robot.sin_addr) != 1) return false;
  const std::string data = std::string(line) + "\n";
  return ::sendto(socket_, data.data(), data.size(), 0, reinterpret_cast<sockaddr*>(&robot),
                  sizeof(robot)) == static_cast<ssize_t>(data.size());
}

std::optional<std::string> SerialTransport::readLine() {
  readAvailable();
  const std::size_t end = pending_.find('\n');
  if (end == std::string::npos) {
    if (pending_.size() > 1024) pending_.clear();  // pas de fin de ligne : déchets
    return std::nullopt;
  }
  std::string line = pending_.substr(0, end);
  pending_.erase(0, end + 1);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return line;
}

ProtocolRobotConfig loadProtocolRobotConfig(const ConfigNode& section) {
  ProtocolRobotConfig cfg;
  cfg.resend_rate = section.getDouble("resend_rate", 5.0, 200.0);
  cfg.ready_timeout = section.getDouble("ready_timeout", 0.0, 10.0);
  return cfg;
}

ProtocolRobot::ProtocolRobot(ITransport& transport, IClock& clock,
                             const ProtocolRobotConfig& config)
    : transport_(transport), clock_(clock), config_(config) {}

ProtocolRobot::~ProtocolRobot() { stop(); }

bool ProtocolRobot::waitReady() {
  const Time deadline = clock_.now() + config_.ready_timeout;
  while (!boot_seen_ && clock_.now() < deadline) {
    poll();
    clock_.sleepUntil(clock_.now() + 0.01);
  }
  return boot_seen_;
}

void ProtocolRobot::transmit() {
  seq_ = (seq_ + 1) & 0xFF;
  sent_at_[seq_] = clock_.now();
  last_sent_ = clock_.now();
  transport_.writeLine(encodeMotor(toPwm(last_.left), toPwm(last_.right), seq_));
}

void ProtocolRobot::send(const WheelCmd& cmd) {
  last_ = cmd;
  transmit();
}

void ProtocolRobot::stop() {
  last_ = {};
  transport_.writeLine(encodeStop());
  last_sent_ = clock_.now();
}

void ProtocolRobot::poll() {
  while (const auto line = transport_.readLine()) {
    const auto msg = parseRobotLine(*line);
    if (!msg) continue;  // ligne corrompue : ignorée
    switch (msg->type) {
      case RobotMessage::Type::Ack:
        last_rtt_ = clock_.now() - sent_at_[msg->seq];
        break;
      case RobotMessage::Type::Watchdog:
        ++watchdog_count_;
        break;
      case RobotMessage::Type::Boot:
        boot_seen_ = true;
        break;
    }
  }
  // Renvoi périodique : sans commande pendant 300 ms, le firmware coupe les moteurs.
  if (clock_.now() - last_sent_ >= 1.0 / config_.resend_rate) {
    transmit();
  }
}
