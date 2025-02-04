import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # Handle image embedding only for test execution phase
    if call.when == "call":
        # Get the pytest_html plugin (if available)
        pytest_html = item.config.pluginmanager.getplugin("html")

        html_content = '<div style="display: flex; gap: 20px; flex-wrap: wrap;">'
        # Extract images tensor from user_properties
        for prop in item.user_properties:
            if prop[0] == "image_tensor" and len(prop) == 2:
                img_tensor = prop[1]
                html_content += '<figure style="margin: 0;">'
            elif prop[0] == "image_tensor" and len(prop) == 3:
                tensor_name = prop[1]
                img_tensor = prop[2]
                html_content += f'<figure style="margin: 0;"><figcaption>{tensor_name}</figcaption>'
            else:
                break
            if img_tensor is not None and pytest_html:
                # Convert tensor to numpy array
                if hasattr(img_tensor, "cpu"):
                    img_tensor = img_tensor.cpu().detach()
                img_np = img_tensor.numpy() if hasattr(img_tensor, "numpy") else np.array(img_tensor)

                # Handle tensor shape (C, H, W) -> (H, W, C)
                if img_np.shape[0] in [1, 3]:
                    img_np = img_np.transpose(1, 2, 0)

                if img_np.dtype == np.float32 and img_np.max() <= 1:
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

                # Convert to base64-encoded image
                buffer = io.BytesIO()
                img = Image.fromarray(img_np)
                if len(img_tensor.shape) == 2:
                    img = img.convert("P")
                    cmap = plt.get_cmap("tab20", 22)
                    colors = cmap(np.arange(22))  # Shape (n_classes, 4) [RGBA]

                    # Convert to 8-bit RGB and flatten into a list
                    palette = (colors[:, :3] * 255).astype(np.uint8).flatten().tolist()

                    # Set Background and Ignore colors
                    palette[:3] = [0, 0, 0]
                    palette.extend([0, 0, 0] * (256 - len(palette) // 3))
                    palette[-3:] = [224, 224, 192]
                    img.putpalette(palette)

                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Add to HTML report using pytest-html's extras system
                html_content += f'<img style="padding: 5px" src="data:image/png;base64,{img_b64}"/></figure>'
            report.extras = [*getattr(report, "extra", []), pytest_html.extras.html(html_content)]


def pytest_html_report_title(report):
    report.title = "PyTorch Image Pipeline Test Report"
