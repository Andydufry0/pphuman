#include "ActionModels.h"
#include <iostream>

// ==================== Shared action engines (fight / fall) ====================

SharedActionEngine &SharedActionEngine::fight() {
    static SharedActionEngine e; return e;
}
SharedActionEngine &SharedActionEngine::fall() {
    static SharedActionEngine e; return e;
}
bool SharedActionEngine::loadOnce(const std::string &path) {
    std::lock_guard<std::mutex> lk(mtx);
    if (eng.ready()) return true;   // already loaded by another channel/switch
    bool ok = eng.load(path);
    if (ok) std::cout << "[SharedAction] loaded " << path << "\n";
    return ok;
}
