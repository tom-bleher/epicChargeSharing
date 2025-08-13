#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <vector>
#include <string>

class EventAction;
class DetectorConstruction;
class G4Step;

// Tracks volume transitions to detect first-contact and sticky
// interactions with pixel-pad and silicon; forwards positions for
// accurate x_hit/y_hit recording.
class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction(EventAction* eventAction, DetectorConstruction* detector);
  virtual ~SteppingAction();
  
  virtual void UserSteppingAction(const G4Step* step);
  
  void Reset();
    
  // Track volume transitions to classify first-contact and sticky pixel/silicon contact flags
  void TrackVolumeInteractions(const G4Step* step);
  
  // Trajectory analysis methods
  G4bool IsValidSiliconHit() const;
  // First-contact classification (true if first entered volume is pixel aluminum)
  G4bool FirstContactIsPixel() const { return fFirstContactVolume == "logicBlock"; }
  
  // Volume-based detection helpers
  G4bool IsInSiliconVolume(const G4Step* step) const;
  G4bool IsPixelHit() const;  // Returns true if track touched pixel volume at any point in the event
    
private:
  EventAction* fEventAction;
  DetectorConstruction* fDetector; // geometry helper
  
  // Per-event classification state
  G4String fFirstContactVolume; // "logicBlock" (pixel) or "logicCube" (silicon) on first boundary entry
  G4bool   fEverTouchedPixel;   // true if any step involved the pixel volume
  G4bool   fEverTouchedSilicon; // true if any step involved the silicon volume
  G4bool   fValidSiliconHit;    // true only if first contact was NOT pixel
  
};

#endif