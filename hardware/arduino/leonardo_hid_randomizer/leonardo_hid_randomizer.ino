// Arduino Leonardo HID "Randomizer" (Friday prep)
// - Bézier curve mouse movement
// - Gaussian delay jitter (Box–Muller)
//
// Upload to: Arduino Leonardo (or Micro) with native USB HID
// Requires: Mouse library (built-in for Leonardo)
//
// Serial protocol (optional, minimal):
//   MOVE dx dy ms\n    -> moves mouse by (dx,dy) over ~ms using a randomized Bézier path
//   JITTER base\n      -> sleeps base milliseconds with gaussian jitter

#include <Mouse.h>

static const long SERIAL_BAUD = 9600;

static float randUnit01() {
  // Avoid 0.0 which breaks log() in Box–Muller
  return (float)random(1, 10001) / 10000.0f;
}

static float randNormal(float mean, float stddev) {
  // Box–Muller transform
  float u1 = randUnit01();
  float u2 = randUnit01();
  float mag = sqrt(-2.0f * log(u1));
  float z0 = mag * cos(2.0f * PI * u2);
  return mean + z0 * stddev;
}

static unsigned long clampUL(long v, unsigned long lo, unsigned long hi) {
  if (v < (long)lo) return lo;
  if (v > (long)hi) return hi;
  return (unsigned long)v;
}

static void jitterDelayGaussian(unsigned long baseMs, float stddevMs, long minJitterMs, long maxJitterMs) {
  long jitter = (long)lround(randNormal(0.0f, stddevMs));
  if (jitter < minJitterMs) jitter = minJitterMs;
  if (jitter > maxJitterMs) jitter = maxJitterMs;

  long total = (long)baseMs + jitter;
  delay(clampUL(total, 0, 600000));
}

struct Pt {
  float x;
  float y;
};

static Pt bezierPoint(const Pt &p0, const Pt &p1, const Pt &p2, const Pt &p3, float t) {
  float u = 1.0f - t;
  float tt = t * t;
  float uu = u * u;
  float uuu = uu * u;
  float ttt = tt * t;

  Pt p;
  p.x = (uuu * p0.x) + (3.0f * uu * t * p1.x) + (3.0f * u * tt * p2.x) + (ttt * p3.x);
  p.y = (uuu * p0.y) + (3.0f * uu * t * p1.y) + (3.0f * u * tt * p2.y) + (ttt * p3.y);
  return p;
}

static void moveMouseBezier(int dx, int dy, unsigned long totalMs) {
  // Steps: tradeoff between smoothness and not looking too uniform
  int steps = 18 + (int)random(0, 10); // 18..27
  if (totalMs < 50) totalMs = 50;

  Pt p0{0.0f, 0.0f};
  Pt p3{(float)dx, (float)dy};

  // Randomized control points (curvature)
  float cx1 = (float)dx * 0.25f + randNormal(0.0f, 6.0f);
  float cy1 = (float)dy * 0.25f + randNormal(0.0f, 6.0f);
  float cx2 = (float)dx * 0.75f + randNormal(0.0f, 6.0f);
  float cy2 = (float)dy * 0.75f + randNormal(0.0f, 6.0f);

  Pt p1{cx1, cy1};
  Pt p2{cx2, cy2};

  Pt prev{0.0f, 0.0f};
  unsigned long stepBase = totalMs / (unsigned long)steps;

  for (int i = 1; i <= steps; i++) {
    float t = (float)i / (float)steps;

    // Slight timing imperfection with gaussian jitter
    // Example: instead of delay(100), do delay(100 + random(-5,15))
    // Here we use gaussian jitter and clamp.
    jitterDelayGaussian(stepBase, stepBase * 0.15f, -5, 15);

    Pt cur = bezierPoint(p0, p1, p2, p3, t);

    float dxF = cur.x - prev.x;
    float dyF = cur.y - prev.y;

    int stepX = (int)lround(dxF);
    int stepY = (int)lround(dyF);

    // Micro-jitter: occasionally add a 0/1 pixel wobble
    if (random(0, 100) < 12) stepX += (int)random(-1, 2);
    if (random(0, 100) < 12) stepY += (int)random(-1, 2);

    Mouse.move(stepX, stepY, 0);
    prev = cur;
  }
}

static bool readLine(String &out) {
  static String buf;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      out = buf;
      buf = "";
      out.trim();
      return true;
    }
    if (c != '\r') {
      buf += c;
    }
  }
  return false;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Mouse.begin();
  randomSeed(analogRead(0));
}

void loop() {
  String line;
  if (!readLine(line)) {
    return;
  }

  if (line.length() == 0) {
    return;
  }

  // MOVE dx dy ms
  if (line.startsWith("MOVE ")) {
    int first = line.indexOf(' ');
    int second = line.indexOf(' ', first + 1);
    int third = line.indexOf(' ', second + 1);

    if (second > 0 && third > 0) {
      int dx = line.substring(first + 1, second).toInt();
      int dy = line.substring(second + 1, third).toInt();
      unsigned long ms = (unsigned long)line.substring(third + 1).toInt();
      moveMouseBezier(dx, dy, ms);
      Serial.println("OK");
      return;
    }
  }

  // JITTER base
  if (line.startsWith("JITTER ")) {
    unsigned long base = (unsigned long)line.substring(7).toInt();
    jitterDelayGaussian(base, 6.0f, -5, 15);
    Serial.println("OK");
    return;
  }

  Serial.println("ERR");
}
