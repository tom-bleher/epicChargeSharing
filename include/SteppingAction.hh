#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <vector>
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
    
  // New methods for aluminum interaction tracking
  void TrackVolumeInteractions(const G4Step* step);
  
  // Enhanced trajectory analysis methods
  G4bool IsValidSiliconHit() const;
    // First-contact classification (true if first entered volume is pixel aluminum)
    G4bool FirstContactIsPixel() const { return fFirstInteractionVolume == "logicBlock"; }
  
  // Volume-based detection methods (replacing IsPosOnPixel logic)
  G4bool IsInSiliconVolume(const G4Step* step) const;
  G4bool IsPixelHit() const;  // Returns true if hit is in pixel volume (aluminum)
    
private:
  EventAction* fEventAction;
  DetectorConstruction* fDetector; // geometry helper
  
  // Enhanced aluminum interaction tracking variables
  G4bool fAluminumInteractionDetected;
  G4bool fSiliconInteractionOccurred;
  G4int fInteractionSequence;
  
  // New trajectory analysis variables
  G4String fFirstInteractionVolume;
  G4bool fAluminumPreContact;
  G4bool fValidSiliconHit;
  G4bool fCurrentHitIsPixel;  // Track if current hit is in pixel volume
  
};

#endif