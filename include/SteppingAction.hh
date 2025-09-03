#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class EventAction;
class G4Step;

// Tracks first-contact and forwards positions.
class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction(EventAction* eventAction);
  ~SteppingAction() override;
  
  void UserSteppingAction(const G4Step* step) override;
  
  void Reset();
    
  // Track volume transitions to classify first-contact only
  void TrackVolumeInteractions(const G4Step* step);
  
  // First-contact classification (true if first entered volume is pixel aluminum)
  G4bool FirstContactIsPixel() const { return fFirstContactVolume == "logicBlock"; }
  
    
private:
  EventAction* fEventAction;
  
  // Per-event classification state
  G4String fFirstContactVolume; // "logicBlock" (pixel) or "logicCube" (silicon) on first boundary entry
  
};

#endif