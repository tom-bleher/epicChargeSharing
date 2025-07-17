#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <vector>
#include <string>

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
  
  // New methods for aluminum interaction tracking
  void ResetInteractionTracking();
  G4bool HasAluminumInteraction() const;
  G4bool IsPureSiliconHit() const;
  void TrackVolumeInteractions(const G4Step* step);
  
  // Enhanced trajectory analysis methods
  G4String GetFirstInteractionVolume() const;
  G4bool HasAluminumPreContact() const;
  G4bool IsValidSiliconHit() const;
  G4int GetInteractionSequence() const;
  
  // Volume-based detection methods (replacing IsPosOnPixel logic)
  G4bool IsInAluminumVolume(const G4Step* step) const;
  G4bool IsInSiliconVolume(const G4Step* step) const;
  G4bool ShouldAccumulateEnergy(const G4Step* step) const;
  G4bool IsPixelHit() const;  // Returns true if hit is in pixel volume (aluminum)
  
  // Comprehensive interaction tracking
  void LogVolumeInteraction(const G4String& volume, G4int sequence);
  std::vector<G4String> GetInteractionHistory() const;
    
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
  
  // Comprehensive interaction history
  std::vector<G4String> fInteractionHistory;
  std::vector<G4int> fInteractionSequenceHistory;
};

#endif