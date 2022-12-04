#include <Wire.h>
#include <OzOLED.h>
#include "position.h"

void setup() {
  OzOled.init();
  print_weight();
  print_CG();
 
}

void loop() {
  LoadCell_Weight();
  CGposition();
}
