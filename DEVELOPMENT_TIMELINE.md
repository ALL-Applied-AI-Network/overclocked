# Team Overclocked development timeline

This retrospective summarizes how Team Overclocked organized the four-week Physical AI Hackathon. The team created weekly plans, reviewed progress and hardware results, and assigned tasks according to each member's experience with robot policies, computer vision, simulation, NVIDIA models, testing, and technical communication. Plans were adjusted as real-cell evidence revealed new constraints.

The Git history preserves the implementation sequence. This document summarizes that work at a project-management level; it does not replace the commit history or claim that every experiment succeeded.

## Week 1 - Define the problem and establish a safe baseline

**Plan:** Understand the CSI cell, select a manufacturing problem, divide the initial integration work, and produce the smallest safe end-to-end robot policy.

**Task assignments:**

- Robot-policy work focused on the CSI client, FANUC motion limits, gripper sequencing, and safe pick/place poses.
- Simulation work focused on the supplied Isaac Sim environment, camera geometry, and reproducible scene layouts.
- Perception work focused on table-camera calibration and a simple visual baseline.
- Documentation work captured setup requirements, constraints, and the initial manufacturing use case.

**Delivered:** A working client baseline, mock-cell validation, initial simulation tooling, and a fixed-layout cube-picking policy. This established that the arm could safely grasp and release a cube before the team added more complex perception.

## Week 2 - Replace fixed coordinates with live perception

**Plan:** Move from a taught-position demonstration to a camera-guided policy that could respond to part location and orientation.

**Task assignments:**

- Vision development added color and silhouette-based cube detection and camera-to-robot coordinate conversion.
- Policy development integrated live detections, multi-frame stability checks, workspace limits, and conservative recovery behavior.
- Simulation and testing work exercised layouts, yaw estimation, and failure cases before additional real-cell runs.
- The team archived policy versions so results could be tied to the behavior that produced them.

**Delivered:** CubeVisionStack v1 and v1.1, followed by controlled position-validation and best-candidate variants. These versions exposed the limits of narrow source regions, expected-count checks, and reliance on submitted positions.

## Week 3 - Diagnose calibration drift and improve robustness

**Plan:** Use physical-cell evidence to determine why policies that worked in one configuration failed after the workcell changed.

**Task assignments:**

- Calibration work compared table-camera detections, wrist views, depth, and gripper offsets.
- Policy work expanded the valid workspace, improved candidate selection, added re-observation, and avoided unsafe blind carries.
- Testing work compared stale and corrected offsets and recorded both successful and unsuccessful outcomes.
- Team reviews converted failures into the next experiment instead of treating every run as a final demonstration.

**Delivered:** Position-validation policies, expanded-region experiments, calibration-learning tools, run modes, and a recovery-oriented control loop. The team identified that the narrow cube grasp window could move substantially after camera or gripper changes.

## Week 4 - Add Physical AI services and validate the final system

**Plan:** Combine the most reliable hardware policy with NVIDIA perception services, a learned grasp check, fallback behavior, and measured comparisons against baselines.

**Task assignments:**

- DGX Spark work integrated OWLv2 detection, Cosmos experiments, C-RADIO features, TensorRT export, and a held-versus-empty grasp classifier.
- Robot-policy work connected table perception, wrist/depth refinement, yaw alignment, grasp checking, placement, and look-again recovery.
- Reliability work verified that classical vision could continue operating when the GPU service was unavailable.
- Evaluation and communication work consolidated sample sizes, baselines, latency, limitations, repository documentation, and the final showcase video.

**Delivered:** The final CSI policy, grasp-verifier service and experiments, GPU-off fallback, recovery demonstrations, and measured hardware results. After recalibration, the final policy placed 12 of 12 cubes across three clean runs; the stale-calibration baseline completed 0 of 3 attempts and produced three controller faults. The team also documented where the system fell short, including small sample sizes, manual recalibration, camera-crop drift, imperfect grasp verification, and limited support for non-cube objects.

## How the team worked

- Weekly plans established the next technical question and the evidence needed to answer it.
- Tasks were assigned by expertise, while integration and real-cell decisions were reviewed as a team.
- Real hardware results took priority over assumptions from simulation.
- Failed trials informed calibration, safety checks, fallback behavior, and the next policy version.
- Code was committed by contribution area so the final implementation, experiments, and version history remained reviewable.

This process produced more than a successful demonstration: it produced a traceable progression from a fixed-position baseline to a perception-driven cell that can re-observe, recover from change, and finish the job.
