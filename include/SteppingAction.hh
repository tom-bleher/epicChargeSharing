#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <string>

class EventAction;
class DetectorConstruction;
class G4Step;

class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction(EventAction* eventAction, DetectorConstruction* detector);
  virtual ~SteppingAction();
  
  virtual void UserSteppingAction(const G4Step* step);
  
  void Reset();
    
  // Getters for hit classification
  G4bool IsValidSiliconHit() const;
  G4bool IsPixelHit() const;
    
private:
  EventAction* fEventAction;
  DetectorConstruction* fDetector;
  
  // Trajectory analysis variables
  G4String fFirstInteractionVolume;
  G4bool fValidSiliconHit;
  G4bool fIsPixelHit;
};

#endif