#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class EventAction;
class DetectorConstruction;

class SteppingAction : public G4UserSteppingAction
{
public:
  // Constructor now also receives DetectorConstruction pointer so we can
  // query geometry helpers (e.g. IsPosOnPixel) during stepping.  This lets us
  // completely ignore any energy deposition for hits whose (x,y) fall inside
  // a pixel footprint, ensuring Gauss-delta calculations never incorporate
  // pixel hits.
  SteppingAction(EventAction* eventAction, DetectorConstruction* detector);
  virtual ~SteppingAction();
  
  virtual void UserSteppingAction(const G4Step* step);
    
private:
  EventAction* fEventAction;
  DetectorConstruction* fDetector; // geometry helper
};

#endif