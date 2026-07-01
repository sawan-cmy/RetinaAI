# Architecture

```text
Client upload (Next.js or FastAPI)
-> src.preprocessing.load_image / crop_black_borders / resize
-> src.quality_check.assess_quality
-> src.models.load_model
   -> Keras CNN artifact: EfficientNet-B0, EfficientNet-B3, ResNet50
   -> fallback sklearn RandomForest baseline
-> src.uncertainty.route_case
-> src.gradcam.generate_keras_gradcam for CNNs
   -> src.gradcam.save_unavailable_explanation for missing/non-CNN models
-> src.inference._recommendation
-> src.report_generator.generate_report
```

Stable Python entry point:

```python
from src.inference import screen_retina_image
result = screen_retina_image("retina.png")
```

Production API:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Frontend:

- `frontend/app/page.tsx`: Dashboard
- `frontend/app/upload/page.tsx`: Upload and connected screening
- `frontend/app/prediction/page.tsx`: Latest prediction
- `frontend/app/gradcam/page.tsx`: Generated explanation artifact
- `frontend/app/metrics/page.tsx`: Evaluation metrics
- `frontend/app/model-comparison/page.tsx`: Model comparison
- `frontend/app/reports/page.tsx`: PDF report viewer
- `frontend/app/history/page.tsx`: Local screening history
- `frontend/app/settings/page.tsx`: Safety settings UI

The random forest remains a fallback baseline. CNN artifacts are selected by model path/config and should be trained with `src.train`.