#include <JANA/JApplication.h>

#include "ChargeSharingReconFactory.h"

#include "extensions/jana/JOmniFactoryGeneratorT.h"

extern "C" {
void InitPlugin(JApplication* app) {
  InitJANAPlugin(app);
  app->Add(new JOmniFactoryGeneratorT<epic::chargesharing::ChargeSharingReconFactory>(
      "ChargeSharingRecon",                 // Tag
      {"SiTrackerHits"},                    // Default input collection (SimTrackerHit)
      {"ChargeSharingTrackerHits"},         // Default output collection (TrackerHit)
      app
  ));
}
}

