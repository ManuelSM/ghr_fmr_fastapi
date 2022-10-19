import base64
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel

from app.engine_ghr import evaluate_ghr
from app.engine_fmr import evaluate_fmr

app = FastAPI()

class ImageBase(BaseModel):
    imageData: str


def get_img(image: ImageBase):
    data = image.imageData
    return base64.b64decode(data)


@app.post("/fmr")
def face_mask_image(image:ImageBase):

    img_output = get_img(image)

    output_info = evaluate_fmr(img_output, target_shape=(260, 260))

    if len(output_info)>1:
        # Mas de una persona detectada
        return {
            "status": 1,
            "description": "Mas de una persona detectada",
        }
    else:
        try:
            if output_info[0][0] == 1:
                # Sin mascarilla
                return {
                    "status": 0,
                    "mask": False,
                    "description": "Foto correcta",
                }
            else: 
                # Con mascarilla 
                return {
                    "status": 0,
                    "mask": True,
                    "description": "Mascarilla detectada",
                }
        except IndexError:
            return{
                "status": 2,
                "description": "No se detectó rostro"
            }


@app.post("/ghr")
def glass_hat_image(image: ImageBase):

    item_hat     = 18
    item_glasses = 6

    img = get_img(image)
    img_ghr = BytesIO(img)

    items_in_image = evaluate_ghr(img_ghr)

    if not items_in_image:
            # Error
            return {
                "status": 1,
                "description": f"Error de parámetro: {items_in_image}"
            }
    else:
        if item_hat in items_in_image and item_glasses in items_in_image:
            # Gafas y/o sombrero detectado
            return {
                "status": 0,
                "glasses": True,
                "hat": True,
                "description": "Gafas y gorra detectados",
            }
        elif item_hat in items_in_image:
            # Gorra detectado 
            return {
                "status": 0,
                "glasses": False,
                "hat": True,
                "description": "Gorra detectada",
            }
        elif item_glasses in items_in_image:
            # Gafas detectadas 
            return {
                "status": 0,
                "glasses": True,
                "hat": False,
                "description": "Gafas detectadas",
            }
        else:
            # Rostro destapado
            return {
                "status": 0,
                "glasses": False,
                "hat": False,
                "description": "Foto correcta",
            }


@app.post("/accessories")
def accessories_image(image: ImageBase):

    img = get_img(image)
    img_ghr = BytesIO(img)
    img_fmr = img

    ghr_items = evaluate_ghr(img_ghr)
    fmr_model_response = evaluate_fmr(img_fmr, target_shape=(260, 260))

    res_ghr = glasses_hat_response(ghr_items)
    res_fmr = face_mask_response(fmr_model_response)

    res_ghr_fmr = {
        **res_ghr,
        **res_fmr
    }

    if res_fmr["description"] == "Foto correcta" and res_ghr["description"] == "Foto correcta":
        
        res_ghr_fmr["description"] = "Ok"

    elif res_fmr["description"] != "Foto correcta" and res_ghr["description"] == "Foto correcta":
        
        res_ghr_fmr["description"] = f"{res_fmr['description']}"
    
    elif res_fmr["description"] == "Foto correcta" and res_ghr["description"] != "Foto correcta":
        
        res_ghr_fmr["description"] = f"{res_ghr['description']}"
    
    else:
        res_ghr_fmr["description"] = f"{res_ghr['description']} {res_fmr['description']}"

    return res_ghr_fmr


def glasses_hat_response(items_in_image: list) -> dict:
    glasses_id = 18
    hat_id = 6

    if len(items_in_image) == 0:
        return {
            "status": 1,
            "description": "No se detectaron elementos en imagen" 
        }

    if glasses_id in items_in_image and hat_id in items_in_image:
        return {
            "status": 0,
            "glasses": True,
            "hat": True,
            "description": "Gafas y gorra detectados",
        }

    if glasses_id in items_in_image:
        return {
            "status": 0,
            "glasses": True,
            "hat": False,
            "description": "Gafas detectadas",
        }

    if hat_id in items_in_image:
        return {
            "status": 0,
            "glasses": False,
            "hat": True,
            "description": "Gorra detectada",
        }

    if glasses_id not in items_in_image and hat_id not in items_in_image:
        return {
            "status": 0,
            "glasses": False,
            "hat": False,
            "description": "Foto correcta"
        }


def face_mask_response(output_info):

    if len(output_info)>1:
        # Mas de una persona detectada
        return {
            "status": 1,
            "description": "Mas de una persona detectada",
        }
    else:
        try:
            if output_info[0][0] == 1:
                # Sin mascarilla
                return {
                    "status": 0,
                    "mask": False,
                    "description": "Foto correcta",
                }
            else: 
                # Con mascarilla 
                return {
                    "status": 0,
                    "mask": True,
                    "description": "Mascarilla detectada",
                }
        except IndexError:
            return{
                "status": 2,
                "description": "No se detectó rostro"
            }