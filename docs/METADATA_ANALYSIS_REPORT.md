# Metadata Storage Analysis Report

## Executive Summary

This report analyzes the current metadata storage implementation in the epicChargeSharing simulation, detailing what is saved, how it's saved, and provides recommendations for improvement.

---

## 1. Current Implementation

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RunAction                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              BuildMetadataPublisher()                        ││
│  │  Collects runtime values from member variables:              ││
│  │  - fGridPixelSize, fGridPixelSpacing, ...                   ││
│  │  - fPosReconModel, fDenominatorMode, ...                    ││
│  │  - Constants::D0, Constants::GAIN, ...                      ││
│  └──────────────────────┬──────────────────────────────────────┘│
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MetadataPublisher                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Struct Categories:                                           ││
│  │  • GridMetadata      (8 fields)                             ││
│  │  • ModelMetadata     (5 fields)                             ││
│  │  • PhysicsMetadata   (4 fields)                             ││
│  │  • NoiseMetadata     (3 fields)                             ││
│  │  • PostProcessMetadata (2 fields)                           ││
│  └──────────────────────┬──────────────────────────────────────┘│
│                         │                                        │
│  CollectEntries() ──────┼───► EntryList (vector<pair<str,str>>) │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WriteEntriesUnlocked()                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ for (key, value) in entries:                                 ││
│  │     TNamed meta(key, value);                                 ││
│  │     meta.Write("", TObject::kOverwrite);                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Storage Method

**Current approach**: `TNamed` objects stored at ROOT file root level

```cpp
// RootIO.cc:467-479
void MetadataPublisher::WriteEntriesUnlocked(TFile* file, const EntryList& entries) {
    file->cd();
    for (const auto& [key, value] : entries) {
        TNamed meta(key.c_str(), value.c_str());
        meta.Write("", TObject::kOverwrite);
    }
    file->Flush();
}
```

**Characteristics**:
- All values stored as strings (no type preservation)
- Objects scattered at file root (not organized in directory or attached to TTree)
- Overwrites existing entries with same key

### 1.3 When Metadata is Written

| Scenario | When Written | Location |
|----------|--------------|----------|
| Single-threaded | End of run | `HandleWorkerEndOfRun()` line 448 |
| Multi-threaded | After merge | `MergeWorkerFilesAndPublishMetadata()` line 575-577 |

---

## 2. What Is Currently Saved

### 2.1 Complete Metadata Field Inventory

| Category | Field | Type | Source | Description |
|----------|-------|------|--------|-------------|
| **Schema** |||||
| | `MetadataSchemaVersion` | String | Hardcoded "2" | Version for backward compatibility |
| **Grid Geometry** |||||
| | `GridPixelSize_mm` | Double | `fGridPixelSize` | Physical pixel size |
| | `GridPixelSpacing_mm` | Double | `fGridPixelSpacing` | Pixel pitch |
| | `GridPixelCornerOffset_mm` | Double | `fGridPixelCornerOffset` | Edge offset |
| | `GridDetectorSize_mm` | Double | `fGridDetSize` | Total detector size |
| | `GridNumBlocksPerSide` | Int | `fGridNumBlocksPerSide` | Pixels per side |
| | `FullGridSide` | Int | `fFullGridSide` | Full grid dimension |
| | `NeighborhoodRadius` | Int | `fGridNeighborhoodRadius` | Neighborhood size |
| **Charge Sharing Model** |||||
| | `ChargeSharingModel` | String | `fPosReconModel` | LogA, LinA, or DPC |
| | `DenominatorMode` | String | `fDenominatorMode` | Neighborhood, RowCol, ChargeBlock |
| | `ChargeSharingLinearBeta_per_um` | Double | `fChargeSharingBeta` | β for LinA/DPC (conditional) |
| | `ChargeSharingPitch_mm` | Double | `fChargeSharingPitch` | Pitch for model |
| **Physics Constants** |||||
| | `ChargeSharingReferenceD0_microns` | Double | `Constants::D0` | d₀ = 1 µm (Tornago Eq.4) |
| | `IonizationEnergy_eV` | Double | `Constants::IONIZATION_ENERGY` | 3.6 eV/pair |
| | `Gain` | Double | `Constants::GAIN` | AC-LGAD gain (20) |
| | `ElementaryCharge_C` | Double | `Constants::ELEMENTARY_CHARGE` | 1.602e-19 C |
| **Noise Model** |||||
| | `NoisePixelGainSigmaMin` | Double | `Constants::PIXEL_GAIN_SIGMA_MIN` | Min gain variation |
| | `NoisePixelGainSigmaMax` | Double | `Constants::PIXEL_GAIN_SIGMA_MAX` | Max gain variation |
| | `NoiseElectronCount` | Double | `Constants::NOISE_ELECTRON_COUNT` | Additive noise (500 e⁻) |
| **Flags** |||||
| | `ChargeSharingEmitDistanceAlpha` | Bool | `fEmitDistanceAlphaMeta` | d_i/α_i stored? |
| | `ChargeSharingFullFractionsEnabled` | Bool | `fStoreFullFractions` | Full grid stored? |
| | `PostProcessFitGaus1DEnabled` | Bool | `Constants::FIT_GAUS_1D` | 1D fit enabled? |
| | `PostProcessFitGaus2DEnabled` | Bool | `Constants::FIT_GAUS_2D` | 2D fit enabled? |

### 2.2 What Is NOT Saved (Missing Metadata)

| Missing Field | Importance | Recommendation |
|---------------|------------|----------------|
| **Simulation Version** | High | Add git commit hash or version string |
| **Geant4 Version** | High | Add `G4VERSION_NUMBER` |
| **ROOT Version** | Medium | Add `ROOT_VERSION_CODE` |
| **Run Timestamp** | High | Add ISO 8601 datetime |
| **Number of Events** | High | Add actual event count |
| **Random Seed** | High | Critical for reproducibility |
| **Primary Particle** | Medium | Type, energy, position config |
| **Detector Material** | Medium | Silicon properties used |
| **CPU/Thread Info** | Low | Number of threads, host info |

---

## 3. Analysis of Current Approach

### 3.1 Strengths

| Aspect | Assessment |
|--------|------------|
| **Simplicity** | Easy to implement and understand |
| **Portability** | TNamed is universally supported in ROOT |
| **Schema versioning** | Allows for future format changes |
| **Thread safety** | Mutex-protected writes |
| **Categorization** | Code-side structs organize parameters logically |

### 3.2 Weaknesses

| Issue | Impact | Severity |
|-------|--------|----------|
| **No type preservation** | All values stored as strings, require parsing | Medium |
| **Flat organization** | All objects at file root, cluttered namespace | Low |
| **Not attached to TTree** | Metadata separate from data it describes | Medium |
| **Conditional writes** | Some fields only written if non-zero/valid | Low |
| **Missing critical fields** | No version, timestamp, seed, event count | High |
| **No validation** | No schema enforcement on read | Low |

---

## 4. Recommendations for Improvement

### 4.1 Immediate Improvements (Low Effort)

#### A. Add Missing Critical Metadata

```cpp
// Add to CollectEntries() in RootIO.cc

// Version information
add("SimulationVersion", PROJECT_VERSION);  // From CMake
add("Geant4Version", std::to_string(G4VERSION_NUMBER));
add("ROOTVersion", std::to_string(ROOT_VERSION_CODE));

// Run information
add("RunTimestamp", GetISOTimestamp());  // New helper function
addInt("TotalEvents", fTotalEvents);     // Track in RunAction
add("RandomSeed", std::to_string(CLHEP::HepRandom::getTheSeed()));

// Primary particle config
add("PrimaryParticle", "proton");  // Or from config
addDouble("PrimaryEnergy_MeV", fPrimaryEnergy);
```

#### B. Add Timestamp Helper

```cpp
// In RootIO.cc
std::string GetISOTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}
```

### 4.2 Medium-Term Improvements

#### A. Use TParameter for Typed Values

```cpp
// Instead of TNamed for all values, use typed parameters:
template<typename T>
void WriteTypedParameter(TFile* file, const std::string& name, T value) {
    file->cd();
    TParameter<T> param(name.c_str(), value);
    param.Write("", TObject::kOverwrite);
}

// Usage:
WriteTypedParameter<Double_t>(file, "Gain", 20.0);
WriteTypedParameter<Int_t>(file, "GridNumBlocksPerSide", 59);
WriteTypedParameter<Bool_t>(file, "FullFractionsEnabled", true);
```

**Benefits**:
- Type preserved in ROOT file
- No string parsing needed on read
- Native support for TBrowser inspection

#### B. Attach Metadata to TTree via UserInfo

```cpp
// In InitializeRootOutputs() after creating tree:
void AttachMetadataToTree(TTree* tree) {
    TList* userInfo = tree->GetUserInfo();

    // Add typed parameters
    userInfo->Add(new TParameter<Double_t>("Gain", Constants::GAIN));
    userInfo->Add(new TParameter<Int_t>("NeighborhoodRadius", fNeighborhoodRadius));

    // Add strings
    userInfo->Add(new TNamed("ChargeSharingModel", "DPC"));
    userInfo->Add(new TNamed("DenominatorMode", "Neighborhood"));
}

// Reading:
TTree* tree = (TTree*)file->Get("Hits");
TList* info = tree->GetUserInfo();
auto* gain = (TParameter<Double_t>*)info->FindObject("Gain");
double gainValue = gain->GetVal();
```

**Benefits**:
- Metadata travels with the TTree
- Clear association between data and its description
- Standard HEP practice (used by ATLAS, CMS, LHCb)

### 4.3 Long-Term Improvements

#### A. Create Dedicated Metadata TTree

```cpp
// Separate metadata tree for complex/variable-length info
TTree* metaTree = new TTree("Metadata", "Simulation Configuration");

// Store as single entry with branches
struct SimConfig {
    char model[32];
    double gain;
    int numPixels;
    // ...
};
SimConfig config;
metaTree->Branch("config", &config, "model[32]/C:gain/D:numPixels/I");
metaTree->Fill();  // Single entry
```

#### B. Use JSON/XML Sidecar

For complex nested configuration, consider writing a JSON sidecar file:

```json
{
  "schema_version": "3.0",
  "simulation": {
    "version": "1.2.0",
    "git_commit": "abc123",
    "timestamp": "2025-12-05T10:30:00Z"
  },
  "detector": {
    "size_mm": 30.0,
    "pixel": {
      "size_mm": 0.1,
      "pitch_mm": 0.5
    }
  },
  "physics": {
    "model": "DPC",
    "denominator_mode": "Neighborhood"
  }
}
```

---

## 5. Implementation Priority

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **P0** | Add timestamp, event count, random seed | Low | High |
| **P0** | Add version strings (sim, G4, ROOT) | Low | High |
| **P1** | Use TParameter for numeric values | Medium | Medium |
| **P1** | Attach core metadata to TTree UserInfo | Medium | Medium |
| **P2** | Add primary particle configuration | Low | Medium |
| **P3** | Create metadata TTree for complex data | High | Low |
| **P3** | JSON sidecar for full configuration dump | Medium | Low |

---

## 6. Example: Reading Current Metadata

```cpp
// C++ example
TFile* f = TFile::Open("epicChargeSharing.root");

// List all metadata
TIter next(f->GetListOfKeys());
TKey* key;
while ((key = (TKey*)next())) {
    if (strcmp(key->GetClassName(), "TNamed") == 0) {
        TNamed* meta = (TNamed*)key->ReadObj();
        std::cout << meta->GetName() << " = " << meta->GetTitle() << std::endl;
    }
}

// Read specific value
TNamed* mode = (TNamed*)f->Get("DenominatorMode");
std::string modeStr = mode ? mode->GetTitle() : "unknown";

// Parse numeric value
TNamed* gain = (TNamed*)f->Get("Gain");
double gainVal = gain ? std::stod(gain->GetTitle()) : 0.0;
```

```python
# Python/uproot example
import uproot

with uproot.open("epicChargeSharing.root") as f:
    # TNamed objects appear as strings in classnames
    for key in f.keys():
        obj = f[key]
        if hasattr(obj, 'title'):  # TNamed-like
            print(f"{key}: {obj.title}")
```

---

## 7. Conclusion

The current metadata implementation is functional but could be enhanced for:

1. **Reproducibility**: Add version, timestamp, and random seed
2. **Type safety**: Use TParameter for numeric values
3. **Organization**: Attach metadata to TTree via UserInfo
4. **Completeness**: Add missing physics configuration

The recommended approach is to implement P0 improvements immediately, followed by P1 changes for better type handling and organization.

---

## References

- [ROOT TFile Documentation](https://root.cern/manual/root_files/)
- [ROOT TTree Class Reference](https://root.cern.ch/doc/master/classTTree.html)
- [ROOT Forum: TNamed/TParameter Best Practices](https://root-forum.cern.ch/t/a-proper-way-to-use-tnamed-tparameter-in-ttree-add/63545)
- [Scikit-HEP TTree Details](https://hsf-training.github.io/hsf-training-scikit-hep-webpage/03-ttree-details/index.html)
