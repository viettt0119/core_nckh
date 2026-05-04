# User Calibration

Per-user calibration is stored here:

```text
models/users/<user_id>/calibration.json
```

Create or update calibration:

```powershell
python calibrate_user.py --user default
```

This follows the EyeMU-style idea of keeping the base gaze model separate from per-user personalization.
