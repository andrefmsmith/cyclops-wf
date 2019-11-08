/* Excitation LED Control v0.1
 * 2019/11/06
*/

// Teensy 2.0 has the LED on pin 11
// Teensy++ 2.0 has the LED on pin 6
// Teensy 3.x / Teensy LC have the LED on pin 13
const int ledPin = 13;
const int bluePin = 6;
const int uvPin = 10;
const int tglobalPin = 2;
int blueState = HIGH;

// the setup() method runs once, when the sketch starts

void setup() {
  // initialize the digital pin as an output.
  pinMode(ledPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  pinMode(uvPin, OUTPUT);
  pinMode(tglobalPin, INPUT);
  attachInterrupt(tglobalPin, tglobalChanged, CHANGE);
}

// called whenever tglobal changes (LOW > HIGH or HIGH > LOW)
void tglobalChanged() {
  int tglobal = digitalRead(tglobalPin);

  // if we have a rising edge of tglobal
  if (tglobal == HIGH) { // start of exposure
    
    // trigger excitation state
    if (blueState == HIGH) {
      digitalWrite(bluePin, HIGH);
      blueState = LOW;
    }
    else {
      digitalWrite(uvPin, HIGH);
      blueState = HIGH;
    }
  }
  else { // end of exposure
    // turn-off excitation
    digitalWrite(uvPin, LOW);
    digitalWrite(bluePin, LOW);
  }
  
  digitalWrite(ledPin, tglobal);
}

// the loop() method runs over and over again,
// as long as the board has power

void loop() {
  
}
