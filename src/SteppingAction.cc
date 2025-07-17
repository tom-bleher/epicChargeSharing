#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"
#include "SimulationLogger.hh"
#include "G4Step.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"

SteppingAction::SteppingAction(EventAction* eventAction, DetectorConstruction* detector)
: G4UserSteppingAction(),
  fEventAction(eventAction),
  fDetector(detector),
  fAluminumInteractionDetected(false),
  fSiliconInteractionOccurred(false),
  fInteractionSequence(0),
  fFirstInteractionVolume("NONE"),
  fAluminumPreContact(false),
  fValidSiliconHit(false),
  fCurrentHitIsPixel(false)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::ResetInteractionTracking()
{
  fAluminumInteractionDetected = false;
  fSiliconInteractionOccurred = false;
  fInteractionSequence = 0;
  fFirstInteractionVolume = "NONE";
  fAluminumPreContact = false;
  fValidSiliconHit = false;
  fCurrentHitIsPixel = false;
  fInteractionHistory.clear();
  fInteractionSequenceHistory.clear();
}

G4bool SteppingAction::HasAluminumInteraction() const
{
  return fAluminumInteractionDetected;
}

G4bool SteppingAction::IsPureSiliconHit() const
{
  return fSiliconInteractionOccurred && !fAluminumInteractionDetected;
}

void SteppingAction::TrackVolumeInteractions(const G4Step* step)
{
  // Get step information for volume tracking
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  // Check if energy was deposited in this step
  G4double edep = step->GetTotalEnergyDeposit();
  if (edep <= 0) return; // No interaction, skip tracking
  
  // Get volume information
  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  
  G4String preVolName = "UNKNOWN";
  G4String postVolName = "UNKNOWN";
  
  if (preVol) preVolName = preVol->GetLogicalVolume()->GetName();
  if (postVol) postVolName = postVol->GetLogicalVolume()->GetName();
  
  // Track interactions chronologically
  fInteractionSequence++;
  
  // Determine the interaction volume (prioritize post-step volume for energy deposit location)
  G4String interactionVolume = postVolName;
  if (interactionVolume == "UNKNOWN" && preVolName != "UNKNOWN") {
    interactionVolume = preVolName;
  }
  
  // Record first interaction volume
  if (fFirstInteractionVolume == "NONE" && interactionVolume != "UNKNOWN") {
    fFirstInteractionVolume = interactionVolume;
    
    // Check if first interaction is with aluminum
    if (interactionVolume == "logicBlock") {
      fAluminumPreContact = true;
    }
  }
  
  // Log this interaction in history
  LogVolumeInteraction(interactionVolume, fInteractionSequence);
  
  // Check for aluminum interaction (pixel electrode volume)
  if (preVolName == "logicBlock" || postVolName == "logicBlock") {
    if (!fAluminumInteractionDetected) {
      fAluminumInteractionDetected = true;
      
      // Log aluminum interaction detection
      SimulationLogger* logger = SimulationLogger::GetInstance();
      if (logger) {
        const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
        G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;
        
        // Log the aluminum interaction with sequence number
        logger->LogInfo("Event " + std::to_string(eventID) + 
                       ": Aluminum interaction detected in sequence " + 
                       std::to_string(fInteractionSequence) + 
                       " (Volume: " + (preVolName == "logicBlock" ? preVolName : postVolName) + ")" +
                       " - First interaction volume: " + fFirstInteractionVolume);
      }
    }
  }
  
  // Check for silicon interaction (detector volume)
  if (preVolName == "logicCube" || postVolName == "logicCube") {
    if (!fSiliconInteractionOccurred) {
      fSiliconInteractionOccurred = true;
      
      // Determine if this is a valid silicon hit (no aluminum pre-contact)
      fValidSiliconHit = !fAluminumPreContact;
      
      // Log silicon interaction detection
      SimulationLogger* logger = SimulationLogger::GetInstance();
      if (logger) {
        const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
        G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;
        
        // Log the silicon interaction with sequence number and validity
        logger->LogInfo("Event " + std::to_string(eventID) + 
                       ": Silicon interaction detected in sequence " + 
                       std::to_string(fInteractionSequence) + 
                       " (Volume: " + (preVolName == "logicCube" ? preVolName : postVolName) + ")" +
                       " - Valid silicon hit: " + (fValidSiliconHit ? "YES" : "NO") +
                       " - First interaction volume: " + fFirstInteractionVolume);
      }
    }
  }
}

// Enhanced trajectory analysis methods
G4String SteppingAction::GetFirstInteractionVolume() const
{
  return fFirstInteractionVolume;
}

G4bool SteppingAction::HasAluminumPreContact() const
{
  return fAluminumPreContact;
}

G4bool SteppingAction::IsValidSiliconHit() const
{
  return fValidSiliconHit;
}

G4bool SteppingAction::IsPixelHit() const
{
  return fCurrentHitIsPixel;
}

G4int SteppingAction::GetInteractionSequence() const
{
  return fInteractionSequence;
}

// Volume-based detection methods (replacing IsPosOnPixel logic)
G4bool SteppingAction::IsInAluminumVolume(const G4Step* step) const
{
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  
  G4String preVolName = preVol ? preVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  
  return (preVolName == "logicBlock" || postVolName == "logicBlock");
}

G4bool SteppingAction::IsInSiliconVolume(const G4Step* step) const
{
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  
  G4String preVolName = preVol ? preVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  
  return (preVolName == "logicCube" || postVolName == "logicCube");
}

G4bool SteppingAction::ShouldAccumulateEnergy(const G4Step* step) const
{
  // Only accumulate energy for silicon volume interactions
  if (!IsInSiliconVolume(step)) {
    return false;
  }
  
  // If this is an aluminum-contaminated event, don't accumulate energy
  if (fAluminumPreContact) {
    return false;
  }
  
  // Only accumulate energy if this is a valid silicon hit
  return fValidSiliconHit || !fAluminumInteractionDetected;
}

// Comprehensive interaction tracking
void SteppingAction::LogVolumeInteraction(const G4String& volume, G4int sequence)
{
  fInteractionHistory.push_back(volume);
  fInteractionSequenceHistory.push_back(sequence);
}

std::vector<G4String> SteppingAction::GetInteractionHistory() const
{
  return fInteractionHistory;
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Track volume interactions for aluminum contamination detection
  TrackVolumeInteractions(step);
  
  // Collect energy deposited in this step while particle travels through detector
  G4double edep = step->GetTotalEnergyDeposit();
  
  // Get step information for tracking
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  // Calculate position of this step (middle of step) - CONSISTENT with AddEdep calculation
  G4ThreeVector stepPos = 0.5 * (prePoint->GetPosition() + postPoint->GetPosition());
  
  // Check if energy was deposited
  if (edep > 0) {
    // Get volume names for debugging
    G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
    G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
    
    G4String preVolName = "UNKNOWN";
    G4String postVolName = "UNKNOWN";
    
    if (preVol) preVolName = preVol->GetLogicalVolume()->GetName();
    if (postVol) postVolName = postVol->GetLogicalVolume()->GetName();
    
    // Track if current hit is in pixel volume (aluminum)
    fCurrentHitIsPixel = IsInAluminumVolume(step);
    
    // ENHANCED VOLUME-BASED DETECTION: Replace broken IsPosOnPixel() logic
    // Use reliable volume-based detection instead of geometric position calculation
    if (!ShouldAccumulateEnergy(step)) {
      // Skip energy accumulation based on volume-based detection
      // This replaces the old IsPosOnPixel() logic with reliable volume detection
      return;
    }

    // Energy deposited inside detector volume with valid silicon interaction
    // Only accumulate if this is a valid, non-contaminated silicon hit
    fEventAction->AddEdep(edep, stepPos);
    
    // Log pixel hit information to SimulationLogger
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      // Get current event ID
      const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
      G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;
      
      // Calculate step length
      G4double stepLength = step->GetStepLength();
      
      // For now, use placeholder pixel indices (they will be calculated properly in EventAction)
      // The real pixel hit determination happens in EventAction::EndOfEventAction
      G4int pixelI = -1;  // Will be determined later
      G4int pixelJ = -1;  // Will be determined later
      
      // Log this step as a hit (individual energy deposition) with volume information
      logger->LogPixelHit(eventID, pixelI, pixelJ, edep, stepPos, stepLength);
      
      // Log volume-based detection information
      logger->LogInfo("Event " + std::to_string(eventID) + 
                     ": Energy accumulated (Volume-based detection)" +
                     " - First interaction: " + fFirstInteractionVolume +
                     " - Valid silicon hit: " + (fValidSiliconHit ? "YES" : "NO") +
                     " - Aluminum pre-contact: " + (fAluminumPreContact ? "YES" : "NO"));
    }
  }
  // Note: Removed excessive debug output for "outside detector" cases
}