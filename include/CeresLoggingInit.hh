#ifndef CERES_LOGGING_INIT_HH
#define CERES_LOGGING_INIT_HH

#include <mutex>

// Thread-safe initialization of Google logging for Ceres
class CeresLoggingInitializer {
public:
    static void InitializeOnce();
    
private:
    static std::once_flag init_flag_;
    static void DoInitialize();
};

#endif // CERES_LOGGING_INIT_HH 