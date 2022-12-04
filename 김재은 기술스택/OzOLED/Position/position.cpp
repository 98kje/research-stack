#include "Arduino.h"
#include "position.h"


void print_weight()
{
  OzOled.printString("1:", 0, 0);
  OzOled.printString("2:", 8, 0);
  OzOled.printString("3:", 0, 2);
  OzOled.printString("Total  : ", 0, 4);
 }

void print_CG()
{
  OzOled.printString("CG     :", 0, 6);
}

void LoadCell_Weight()
{
  OzOled.printString("0.0", 2, 0);
  OzOled.printString("kg",5,0);
  OzOled.printString("0.0", 10, 0);
  OzOled.printString("kg",13,0);
  OzOled.printString("0.0", 2, 2);
  OzOled.printString("kg",5,2);
  OzOled.printString("0,0", 10 ,4);
  OzOled.printString("kg",13,4);
}

void CGposition()
{
  OzOled.printString("x , y", 10 ,6);
}
