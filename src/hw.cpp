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

