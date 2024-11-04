from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import coremltools as ct

# Загрузка модели
model_path = '/Users/anymacstore/Documents/MyImageClassifier.mlmodel'
model = ct.models.MLModel(model_path)

# Создание FastAPI приложения
app = FastAPI()

# Получение названия входного параметра
input_name = model.get_spec().description.input[0].name

# Функция для подготовки изображения
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    return image

# Обработчик маршрута для предсказания
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение содержимого загруженного файла
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Подготовка изображения
        processed_image = preprocess_image(image)

        # Прогноз модели
        prediction = model.predict({input_name: processed_image})

        # Возврат результата классификации
        return JSONResponse(content=prediction)
    except Exception as e:
        # Вывод ошибки в консоль для отладки
        print(f"Ошибка: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Запуск приложения с uvicorn, если файл запускается как основной
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
