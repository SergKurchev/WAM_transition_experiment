# Camera Setup for WAM Inference

## Camera Placement in Isaac Sim

The World Action Model (WAM) expects visual observations from a camera mounted on or near the robot.

### Recommended Camera Position for G1 Robot

**Position:** Head-mounted or chest-mounted perspective
- **Location:** Above and in front of the robot's body
- **Height:** ~1.5m (robot head level) for forward-looking view
- **Orientation:** Looking downward at 45° angle to capture arms and torso
- **Offset from robot:** 0.3m forward, 0.2m up

### Expected Frame Specifications

- **Resolution:** 512×512 or 1024×768 (adjust based on model input)
- **Frame Rate:** 10 Hz (synchronized with control loop)
- **Format:** RGB PNG or similar
- **Field of View:** 60-90° (wide enough to see arm extensions)

### Isaac Sim Configuration

1. **Add Camera in USD:**
   ```
   def Prim "World"
   {
       def Xform "G1" (references = @./g1.usd@)
       {
           def Camera "front_camera"
           {
               float focalLength = 24
               float horizontalAperture = 20.955
               float verticalAperture = 11.7622
           }
       }
   }
   ```

2. **Capture Frame:**
   - Use Isaac Sim's `omni.syntheticdata` module to capture RGB from camera
   - Write frame to `/tmp/isaac_frame.png` every control loop iteration
   - The WAM recorder will automatically copy this to `media/isaac_frames/`

3. **Verify in Visualization:**
   - Check `http://localhost:6081` (noVNC tunnel)
   - Camera view should show G1 robot with arms extending forward

### Current Status

- [x] Input frame recording (RobotState as text)
- [x] Command logging (velocity outputs to CSV)
- [ ] Isaac Sim camera frame capture (integrate with Isaac Sim rendering)
- [ ] Video generation from frames (post-processing step)

### Testing Camera Feed

```bash
# On server, check if camera frame is being written
watch -n 1 'ls -lh /tmp/isaac_frame.png'

# On local, copy and check frame
bash scripts/copy-media-from-server.sh --frames-only
file media_local/isaac_frames/isaac_000000.png
```

### Integration with original Paper

The original UnifoLM paper uses visual observations for context encoding. Our implementation:
1. Captures camera frames at 10 Hz during inference
2. Saves frames to `media/isaac_frames/` for later analysis
3. Can be used for post-hoc video generation
4. Enables debugging and visualization of robot behavior

