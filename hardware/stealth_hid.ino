/*
 * Arduino HID Bridge for EVE Online Zombie Node
 * Defines Serial -> Keyboard + Mouse mappings.
 *
 * Serial protocol:
 *   PING                  -> PONG
 *   CAPS                  -> CAPS:KEYBOARD,MOUSE,MOVCLK_V1
 *   ENTER / TAB / ALT+M... -> keyboard macros
 *   PRESS:x               -> press+release key char
 *   TYPE:text             -> type text
 *   MOV:x,y               -> move pointer to approx absolute pixel (best-effort)
 *   CLK                   -> left click
 */

#include <Keyboard.h>
#include <Mouse.h>

void processCommand(String cmd);

// Configure the screen size used by MOV:x,y.
// This is a best-effort absolute move implemented via relative HID moves.
// If your display is different, update these constants and re-upload.
static const int SCREEN_W = 1920;
static const int SCREEN_H = 1080;

static int g_cursor_x = 0;
static int g_cursor_y = 0;
static bool g_cursor_homed = false;

static void mouseHome() {
  // Slam the cursor to top-left by repeatedly moving up-left.
  // This is the only way to approximate absolute positioning with the built-in Mouse library.
  for (int i = 0; i < 70; i++) {
    Mouse.move(-127, -127, 0);
    delay(2);
  }
  g_cursor_x = 0;
  g_cursor_y = 0;
  g_cursor_homed = true;
}

static int clampInt(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static void mouseMoveTo(int x, int y) {
  if (!g_cursor_homed) {
    mouseHome();
  }
  x = clampInt(x, 0, SCREEN_W - 1);
  y = clampInt(y, 0, SCREEN_H - 1);

  int dx = x - g_cursor_x;
  int dy = y - g_cursor_y;

  // Smaller steps reduce the impact of pointer acceleration.
  const int STEP = 60;
  while (dx != 0 || dy != 0) {
    int stepX = clampInt(dx, -STEP, STEP);
    int stepY = clampInt(dy, -STEP, STEP);
    if (stepX == 0 && dx != 0) stepX = (dx > 0) ? 1 : -1;
    if (stepY == 0 && dy != 0) stepY = (dy > 0) ? 1 : -1;

    Mouse.move(stepX, stepY, 0);
    g_cursor_x += stepX;
    g_cursor_y += stepY;

    dx = x - g_cursor_x;
    dy = y - g_cursor_y;
    delay(random(3, 9));
  }
}

void setup() {
  Serial.begin(9600);
  Keyboard.begin();
  Mouse.begin();
  
  // Wait for Serial to establish
  while (!Serial) {
    ; // wait
  }
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

void processCommand(String cmd) {
  // Protocol: "KEY:c" or "MOD:ctrl+c"

  // Handshake / health-check
  if (cmd == "PING") {
    Serial.println("PONG");
  }

  // Capability discovery
  else if (cmd == "CAPS") {
    Serial.println("CAPS:KEYBOARD,MOUSE,MOVCLK_V1");
  }
  
  // Simple single key press
  else if (cmd.startsWith("PRESS:")) {
    char key = cmd.substring(6)[0];
    Keyboard.press(key);
    delay(random(80, 150)); // Hardware Jitter
    Keyboard.release(key);
    Serial.println("OK");
  }
  
  // Special Keys
  else if (cmd == "ENTER") {
    Keyboard.press(KEY_RETURN);
    delay(random(80, 150));
    Keyboard.release(KEY_RETURN);
    Serial.println("OK");
  }
  else if (cmd == "TAB") {
    Keyboard.press(KEY_TAB);
    delay(random(80, 150));
    Keyboard.release(KEY_TAB);
    Serial.println("OK");
  }
  
  // Combos
  else if (cmd == "ALT+M") {
    Keyboard.press(KEY_LEFT_ALT);
    delay(random(20, 50));
    Keyboard.press('m');
    delay(random(80, 150));
    Keyboard.release('m');
    Keyboard.release(KEY_LEFT_ALT);
    Serial.println("OK");
  }
  else if (cmd == "ALT+C") {
    Keyboard.press(KEY_LEFT_ALT);
    delay(random(20, 50));
    Keyboard.press('c');
    delay(random(80, 150));
    Keyboard.release('c');
    Keyboard.release(KEY_LEFT_ALT);
    Serial.println("OK");
  }
  else if (cmd == "SHIFT+S") {
    Keyboard.press(KEY_LEFT_SHIFT);
    delay(random(20, 50));
    Keyboard.press('s');
    delay(random(80, 150));
    Keyboard.release('s');
    Keyboard.release(KEY_LEFT_SHIFT);
    Serial.println("OK");
  }
  else if (cmd == "CTRL+SHIFT+F9") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press(KEY_LEFT_SHIFT);
    delay(random(20, 50));
    Keyboard.press(KEY_F9);
    delay(random(80, 150));
    Keyboard.release(KEY_F9);
    Keyboard.release(KEY_LEFT_SHIFT);
    Keyboard.release(KEY_LEFT_CTRL);
    Serial.println("OK");
  }
  
  // Text Entry (for search fields)
  else if (cmd.startsWith("TYPE:")) {
    String text = cmd.substring(5);
    Keyboard.print(text);
    Serial.println("OK");
  }

  // Mouse absolute move (best-effort) + click
  else if (cmd.startsWith("MOV:")) {
    int comma = cmd.indexOf(',', 4);
    if (comma > 4) {
      int x = cmd.substring(4, comma).toInt();
      int y = cmd.substring(comma + 1).toInt();
      mouseMoveTo(x, y);
      Serial.println("OK");
    } else {
      Serial.println("ERR");
    }
  }
  else if (cmd == "CLK") {
    Mouse.press(MOUSE_LEFT);
    delay(random(40, 90));
    Mouse.release(MOUSE_LEFT);
    Serial.println("OK");
  }

  else {
    // Unknown command
    Serial.println("ERR");
  }
  
  delay(random(50, 100)); // Reset delay
}
