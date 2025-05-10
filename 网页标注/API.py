from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model import load_model, generate_excel, mark_image  # 确保导入model.py
import os
import shutil
import uuid
import webbrowser
import uvicorn

app = FastAPI()

# 确保目录存在
STATIC_DIR = "static"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
for directory in [STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 全局加载模型以避免重复加载
try:
    model = load_model()
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise

# 提供 HTML 页面
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found in static directory")

# 图像标注端点
@app.post("/annotate")
async def annotate_image(
    file: UploadFile = File(...),
    target_class: int = Form(...,
                             description="Target class index (0: MildDemented, 1: ModerateDemented, "
                                         "2: NonDemented, 3: VeryMildDemented)")
):
    image_path = None
    exc_path = None
    output_image_path = None
    try:
        # 验证目标类别
        if target_class not in [0, 1, 2, 3]:
            raise HTTPException(status_code=400, detail="Target class must be 0, 1, 2, or 3")

        # 验证文件扩展名
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["png", "jpg", "jpeg"]:
            raise HTTPException(status_code=400, detail="Only PNG, JPG, or JPEG files are supported")

        # 保存上传的图像
        image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{file_extension}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 设置输出路径
        exc_path = os.path.join(OUTPUT_DIR, f"output_{uuid.uuid4()}.xlsx")
        output_image_path = os.path.join(OUTPUT_DIR, f"annotated_{uuid.uuid4()}.png")

        # 生成 Excel 文件
        generate_excel(image_path, exc_path, target_class)

        # 验证 Excel 文件是否生成
        if not os.path.exists(exc_path):
            raise HTTPException(status_code=500, detail="Failed to generate Excel file")

        # 生成标注图像
        mark_image(exc_path, image_path, output_image_path)

        # 验证标注图像是否生成
        if not os.path.exists(output_image_path):
            raise HTTPException(status_code=500, detail="Failed to generate annotated image")

        # 将输出文件移动到 static 目录
        static_excel_path = os.path.join(STATIC_DIR, os.path.basename(exc_path))
        static_image_path = os.path.join(STATIC_DIR, os.path.basename(output_image_path))
        shutil.move(exc_path, static_excel_path)
        shutil.move(output_image_path, static_image_path)

        # 返回文件路径和统计信息
        return {
            "excel_file": f"/static/{os.path.basename(static_excel_path)}",
            "annotated_image": f"/static/{os.path.basename(static_image_path)}",
            "message": "Annotation completed successfully"
        }

    except Exception as e:
        # 清理临时文件
        for path in [image_path, exc_path, output_image_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# 自动启动服务器并打开浏览器
if __name__ == "__main__":
    url = "http://localhost:8000"
    webbrowser.open(url)
    uvicorn.run(app, host="0.0.0.0", port=8000)