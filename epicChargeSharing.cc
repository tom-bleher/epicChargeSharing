#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "G4MTRunManager.hh"
#include "G4RunManager.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "RuntimeConfig.hh"

#include <filesystem>

namespace
{
struct ProgramOptions {
    G4bool isBatch{false};
    G4String macroFile{};
    G4int requestedThreads{-1};
};

#ifdef _WIN32
void SetEnv(const std::string& key, const std::string& value)
{
    _putenv_s(key.c_str(), value.c_str());
}

void UnsetEnv(const std::string& key)
{
    _putenv_s(key.c_str(), "");
}
#else
void SetEnv(const std::string& key, const std::string& value)
{
    ::setenv(key.c_str(), value.c_str(), 1);
}

void UnsetEnv(const std::string& key)
{
    ::unsetenv(key.c_str());
}
#endif

class EnvironmentGuard
{
public:
    EnvironmentGuard() = default;

    EnvironmentGuard(std::string key, std::string value)
    {
        Apply(std::move(key), std::move(value));
    }

    ~EnvironmentGuard()
    {
        Restore();
    }

    void Apply(std::string key, std::string value)
    {
        Restore();
        fKey = std::move(key);
        const char* current = std::getenv(fKey.c_str());
        if (current) {
            fPrevious.emplace(current);
        }
        SetEnv(fKey, value);
        fApplied = true;
    }

    void Restore()
    {
        if (!fApplied) {
            return;
        }
        if (fPrevious) {
            SetEnv(fKey, *fPrevious);
        } else {
            UnsetEnv(fKey);
        }
        fApplied = false;
        fPrevious.reset();
    }

    EnvironmentGuard(const EnvironmentGuard&) = delete;
    EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;
    EnvironmentGuard(EnvironmentGuard&& other) noexcept
    {
        *this = std::move(other);
    }
    EnvironmentGuard& operator=(EnvironmentGuard&& other) noexcept
    {
        if (this != &other) {
            Restore();
            fKey = std::move(other.fKey);
            fPrevious = std::move(other.fPrevious);
            fApplied = other.fApplied;
            other.fApplied = false;
        }
        return *this;
    }

private:
    std::string fKey;
    std::optional<std::string> fPrevious;
    bool fApplied{false};
};

ProgramOptions ParseArguments(int argc, char** argv)
{
    ProgramOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-m") {
            opts.isBatch = true;
            if (i + 1 < argc) {
                opts.macroFile = argv[++i];
            } else {
                throw std::runtime_error("Error: -m requires a filename argument");
            }
        } else if (arg == "-t") {
            if (i + 1 < argc) {
                opts.requestedThreads = std::atoi(argv[++i]);
                if (opts.requestedThreads <= 0) {
                    throw std::runtime_error("Error: Invalid number of threads");
                }
            } else {
                throw std::runtime_error("Error: -t requires a number argument");
            }
        } else {
            throw std::runtime_error("Error: Unknown option: " + arg);
        }
    }

    return opts;
}

G4RunManager* CreateRunManager(const ProgramOptions& opts)
{
#ifdef G4MULTITHREADED
    if (opts.requestedThreads != 1) {
        auto* mtRunManager = new G4MTRunManager;

        G4int nThreads = 0;
        if (opts.requestedThreads > 0) {
            nThreads = opts.requestedThreads;
        } else {
            nThreads = G4Threading::G4GetNumberOfCores();
        }

        const G4int maxThreads = G4Threading::G4GetNumberOfCores();
        if (nThreads > maxThreads) {
            nThreads = maxThreads;
        }

        mtRunManager->SetNumberOfThreads(nThreads);
        G4cout << "[Main] Multithreaded mode with " << nThreads << " / " << maxThreads
               << " cores (" << (opts.isBatch ? "batch" : "interactive") << ")" << G4endl;

        return mtRunManager;
    }
#endif
    G4cout << "[Main] Single-threaded mode (" << (opts.isBatch ? "batch" : "interactive")
           << ")" << G4endl;
    return new G4RunManager;
}

void ConfigureInitialization(G4RunManager* runManager, DetectorConstruction* detector)
{
    runManager->SetUserInitialization(detector);
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserInitialization(new ActionInitialization(detector));
}

bool ExecuteBatchMode(G4RunManager* runManager, const ProgramOptions& opts)
{
    (void)runManager;

    G4UImanager* uiManager = G4UImanager::GetUIpointer();
    G4cout << "[Batch] Executing macro '" << opts.macroFile << "'" << G4endl;
    G4String command = "/control/execute " + opts.macroFile;
    const G4int status = uiManager->ApplyCommand(command);
    if (status != 0) {
        G4cerr << "[Batch] Macro execution failed: " << opts.macroFile << G4endl;
        return false;
    }
    return true;
}

void ExecuteInteractiveMode(G4RunManager* runManager, int argc, char** argv)
{
    (void)runManager;

    auto ui = std::make_unique<G4UIExecutive>(argc, argv);
    auto visManager = std::make_unique<G4VisExecutive>();
    visManager->Initialize();

    G4UImanager* uiManager = G4UImanager::GetUIpointer();

    try {
        if (std::filesystem::exists("vis.mac")) {
            uiManager->ApplyCommand("/control/execute vis.mac");
        } else if (std::filesystem::exists("macros/vis.mac")) {
            uiManager->ApplyCommand("/control/execute macros/vis.mac");
        }
    } catch (...) {
        // ignore missing vis macro
    }

    ui->SessionStart();
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const ProgramOptions opts = ParseArguments(argc, argv);
        G4cout << "[Main] Starting " << (opts.isBatch ? "batch" : "interactive")
               << " session" << G4endl;

        EnvironmentGuard qtGuard;
        EnvironmentGuard uiGuard;

        if (opts.isBatch) {
            qtGuard.Apply("QT_QPA_PLATFORM", "offscreen");
            uiGuard.Apply("EPIC_INTERACTIVE_UI", "0");
        } else {
            uiGuard.Apply("EPIC_INTERACTIVE_UI", "1");
        }

        // Initialize runtime configuration (registers /ecs/ macro commands).
        // Must be created before RunManager so macro commands are available
        // when macros are parsed.
        auto& runtimeConfig = ECS::RuntimeConfig::Instance();
        (void)runtimeConfig;  // Suppress unused variable warning

        auto* detector = new DetectorConstruction();
        std::unique_ptr<G4RunManager> runManager(CreateRunManager(opts));
        ConfigureInitialization(runManager.get(), detector);

        if (opts.isBatch) {
            const bool success = ExecuteBatchMode(runManager.get(), opts);
            if (!success) {
                return 1;
            }
        } else {
            ExecuteInteractiveMode(runManager.get(), argc, argv);
        }

        return 0;
    } catch (const std::exception& ex) {
        G4cerr << "[Main] Unhandled exception: " << ex.what() << G4endl;
        return 1;
    }
}
