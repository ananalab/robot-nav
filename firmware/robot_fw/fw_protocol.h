// Protocole série robot <-> PC, côté firmware (§6.7).
// C++ pur, sans Arduino : ce même fichier est compilé dans les tests du PC
// (tests/test_hw.cpp) et par le simulateur de firmware (SimFirmware).
//
// Une trame par ligne, terminée par '\n' :
//   PC -> robot : M,<l>,<r>,<seq>*CK   moteurs, l et r dans [-255, 255], seq dans [0, 255]
//                 S*CK                 arrêt immédiat
//                 P,<seq>*CK           ping
//   robot -> PC : A,<seq>*CK           acquittement de M et P
//                 W*CK                 watchdog déclenché, moteurs arrêtés
//                 B*CK                 démarrage terminé
// CK = 2 chiffres hexadécimaux MAJUSCULES = XOR de tous les octets avant '*'.
#ifndef ROBOT_FW_PROTOCOL_H
#define ROBOT_FW_PROTOCOL_H

#include <stdint.h>
#include <stdio.h>

#define FW_LINE_MAX 32  // taille du buffer de réception (octets, '\0' compris)

enum FwCommandType { FW_NONE = 0, FW_MOTOR, FW_STOP, FW_PING };

struct FwCommand {
  FwCommandType type;
  int left;   // PWM gauche [-255, 255]
  int right;  // PWM droite [-255, 255]
  int seq;    // [0, 255]
};

// XOR de n octets.
inline uint8_t fwChecksum(const char* s, int n) {
  uint8_t ck = 0;
  for (int i = 0; i < n; ++i) ck ^= static_cast<uint8_t>(s[i]);
  return ck;
}

inline int fwHexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;  // minuscules refusées : le format impose des majuscules
}

// Lit un entier signé décimal entre p et le séparateur sep ; avance p. false si invalide.
inline bool fwReadInt(const char*& p, char sep, int min_value, int max_value, int& out) {
  bool negative = false;
  if (*p == '-') {
    negative = true;
    ++p;
  }
  if (*p < '0' || *p > '9') return false;
  long value = 0;
  int digits = 0;
  while (*p >= '0' && *p <= '9') {
    value = value * 10 + (*p - '0');
    if (++digits > 4) return false;  // évite tout débordement
    ++p;
  }
  if (*p != sep) return false;
  ++p;
  if (negative) value = -value;
  if (value < min_value || value > max_value) return false;
  out = static_cast<int>(value);
  return true;
}

// Analyse une ligne SANS le '\n'. Renvoie false (et type FW_NONE) si la trame est invalide.
inline bool fwParseLine(const char* line, FwCommand& cmd) {
  cmd.type = FW_NONE;
  cmd.left = cmd.right = cmd.seq = 0;
  // Position de '*' et contrôle de la longueur.
  int star = -1, len = 0;
  while (line[len] != '\0') {
    if (line[len] == '*' && star < 0) star = len;
    if (++len >= FW_LINE_MAX) return false;
  }
  if (star < 1 || len != star + 3) return false;
  const int hi = fwHexDigit(line[star + 1]);
  const int lo = fwHexDigit(line[star + 2]);
  if (hi < 0 || lo < 0 || fwChecksum(line, star) != static_cast<uint8_t>(hi * 16 + lo)) return false;

  const char* p = line + 1;
  switch (line[0]) {
    case 'S':
      if (star != 1) return false;
      cmd.type = FW_STOP;
      return true;
    case 'P':
      if (*p++ != ',') return false;
      if (!fwReadInt(p, '*', 0, 255, cmd.seq)) return false;
      cmd.type = FW_PING;
      return true;
    case 'M':
      if (*p++ != ',') return false;
      if (!fwReadInt(p, ',', -255, 255, cmd.left)) return false;
      if (!fwReadInt(p, ',', -255, 255, cmd.right)) return false;
      if (!fwReadInt(p, '*', 0, 255, cmd.seq)) return false;
      cmd.type = FW_MOTOR;
      return true;
    default:
      return false;
  }
}

// Écrit "<body>*CK" dans out (taille >= FW_LINE_MAX). Renvoie la longueur.
inline int fwFrame(const char* body, char* out) {
  int n = 0;
  while (body[n] != '\0' && n < FW_LINE_MAX - 4) {
    out[n] = body[n];
    ++n;
  }
  const uint8_t ck = fwChecksum(out, n);
  const char* hex = "0123456789ABCDEF";
  out[n] = '*';
  out[n + 1] = hex[ck >> 4];
  out[n + 2] = hex[ck & 0x0F];
  out[n + 3] = '\0';
  return n + 3;
}

#endif  // ROBOT_FW_PROTOCOL_H
