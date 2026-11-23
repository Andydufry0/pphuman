#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <QCoreApplication>

// crash handler: symbolized stack trace to stderr on fatal exception
// (industrial deployments are headless — crashes must leave evidence)
#pragma comment(lib, "dbghelp.lib")

static void printTrace(EXCEPTION_POINTERS *e) {
    fprintf(stderr, "\n===== CRASH: code=0x%08lX addr=%p thread=%lu =====\n",
            e->ExceptionRecord->ExceptionCode,
            e->ExceptionRecord->ExceptionAddress,
            GetCurrentThreadId());
    fflush(stderr);

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    HANDLE proc = GetCurrentProcess();
    if (!SymInitialize(proc, NULL, TRUE))
        fprintf(stderr, "SymInitialize failed\n"), fflush(stderr);

    CONTEXT *ctx = e->ContextRecord;
    STACKFRAME64 frame;
    memset(&frame, 0, sizeof(frame));
#ifdef _WIN64
    DWORD machine = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = ctx->Rip;
    frame.AddrFrame.Offset = ctx->Rbp;
    frame.AddrStack.Offset = ctx->Rsp;
#else
    DWORD machine = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset = ctx->Eip;
    frame.AddrFrame.Offset = ctx->Ebp;
    frame.AddrStack.Offset = ctx->Esp;
#endif
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Mode = AddrModeFlat;

    for (int i = 0; i < 30; ++i) {
        if (!StackWalk64(machine, proc, GetCurrentThread(), &frame, ctx,
                         NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL))
            break;
        DWORD64 pc = frame.AddrPC.Offset;
        if (!pc) break;

        char buf[sizeof(SYMBOL_INFO) + 256];
        auto *sym = reinterpret_cast<SYMBOL_INFO *>(buf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        if (SymFromAddr(proc, pc, &disp, sym))
            fprintf(stderr, "  #%02d %s +0x%llu\n", i, sym->Name,
                    (unsigned long long)disp);
        else
            fprintf(stderr, "  #%02d 0x%llX\n", i, (unsigned long long)pc);
    }
    fflush(stderr);
    Sleep(500);   // let stderr flush
}

static LONG WINAPI crashHandler(EXCEPTION_POINTERS *e) {
    printTrace(e);
    return EXCEPTION_EXECUTE_HANDLER;
}

void installCrashHandler() {
    SetUnhandledExceptionFilter(crashHandler);
}
