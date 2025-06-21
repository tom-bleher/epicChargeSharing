#include "CeresLoggingInit.hh"
#include "glog/logging.h"

// Static member definitions
std::once_flag CeresLoggingInitializer::init_flag_;

void CeresLoggingInitializer::InitializeOnce() {
    std::call_once(init_flag_, DoInitialize);
}

void CeresLoggingInitializer::DoInitialize() {
    google::InitGoogleLogging("epicChargeSharing");
    // Completely suppress all Ceres logging
    FLAGS_logtostderr = false;
    FLAGS_minloglevel = 4; // Suppress all messages including FATAL
    FLAGS_stderrthreshold = 4;
    FLAGS_log_dir = "/tmp";
} 