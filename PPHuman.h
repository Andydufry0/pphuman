#pragma once
// Compatibility aggregate header.
// New code should include only the model header it needs:
//   PpDetModel.h  - PPYOLOE detector (MOT / smoking)
//   KptModel.h    - HRNet keypoint batch inference
//   FightModel.h  - ppTSM fight classifier
//   FallModel.h   - STGCN fall classifier
#include "ModelTypes.h"
#include "PpDetModel.h"
#include "KptModel.h"
#include "ActionModels.h"
#include "FightModel.h"
#include "FallModel.h"
