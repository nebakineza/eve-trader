/*
 * Arduino HID Bridge for EVE Online Zombie Node
 * Defines Serial -> Keyboard mappings
 */

#include <Keyboard.h>

void setup() {
  Serial.begin(9600);
  Keyboard.begin();
  
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
  
  // Simple single key press
  if (cmd.startsWith("PRESS:")) {
    char key = cmd.substring(6)[0];
    Keyboard.press(key);
    delay(random(80, 150)); // Hardware Jitter
    Keyboard.release(key);
  }
  
  // Special Keys
  else if (cmd == "ENTER") {
    Keyboard.press(KEY_RETURN);
    delay(random(80, 150));
    Keyboard.release(KEY_RETURN);
  }
  else if (cmd == "TAB") {
    Keyboard.press(KEY_TAB);
    delay(random(80, 150));
    Keyboard.release(KEY_TAB);
  }
  
  // Combos
  else if (cmd == "ALT+M") {
    Keyboard.press(KEY_LEFT_ALT);
    delay(random(20, 50));
    Keyboard.press('m');
    delay(random(80, 150));
    Keyboard.release('m');
    Keyboard.release(KEY_LEFT_ALT);
  }
  else if (cmd == "SHIFT+S") {
    Keyboard.press(KEY_LEFT_SHIFT);
    delay(random(20, 50));
    Keyboard.press('s');
    delay(random(80, 150));
    Keyboard.release('s');
    Keyboard.release(KEY_LEFT_SHIFT);
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
  }
  
  // Text Entry (for search fields)
  else if (cmd.startsWith("TYPE:")) {
    String text = cmd.substring(5);
    Keyboard.print(text);
  }
  
  delay(random(50, 100)); // Reset delay
}
