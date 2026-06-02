import os
import gradio as gr
from transformers import pipeline

# Ruta local donde se guardo el modelo y tokenizador
MODEL_PATH = "./modelo_ner_albert"

# Inicializamos el pipeline especializado en Reconocimiento de Entidades Nombradas (NER)
# Usamos aggregation_strategy="simple" para que junte los subtokens de ALBERT en palabras completas
try:
    ner_pipeline = pipeline(
        "ner", 
        model=MODEL_PATH, 
        tokenizer=MODEL_PATH, 
        aggregation_strategy="simple"
    )
except Exception as e:
    # En caso de que se ejecute en Spaces y la carpeta esté en la raíz
    ner_pipeline = pipeline(
        "ner", 
        model=".", 
        aggregation_strategy="simple"
    )

def merge_and_highlight_ner(text):
    if not text.strip():
        return {"text": text, "entities": []}
        
    # Ejecutar la inferencia del modelo
    ner_results = ner_pipeline(text)
    
    entities = []
    for entity in ner_results:
        # Mapeamos las etiquetas técnicas del dataset a nombres legibles para la UI
        label_mapping = {
            "PER": "Persona",
            "LOC": "Lugar / Ubicación",
            "ORG": "Organización",
            "MISC": "Misceláneo"
        }
        
        # Obtenemos la categoría limpia quitando prefijos si los hay
        clean_label = entity['entity_group']
        friendly_label = label_mapping.get(clean_label, clean_label)
        
        # Estructura requerida por gr.HighlightedText: (inicio_char, fin_char, etiqueta)
        entities.append({
            "entity": friendly_label,
            "start": entity['start'],
            "end": entity['end']
        })
        
    return {"text": text, "entities": entities}

# Construcción de la Interfaz Web Interactiva con Gradio Blocks
with gr.Blocks(title="NER con ALBERT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏷️ Reconocimiento de Entidades Nombradas (NER) con ALBERT")
    gr.Markdown(
        "Prototipo académico individual desarrollado para la **Práctica 5**. "
        "Este modelo utiliza la arquitectura optimizada **ALBERT-base-v2** (~11M de parámetros) "
        "finitamente entrenada sobre el dataset *CoNLL-2003*."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=5, 
                placeholder="Escribe un enunciado en inglés (ej. nombres de personas, lugares o empresas)...", 
                label="Texto de Entrada"
            )
            submit_btn = gr.Button("Detectar Entidades", variant="primary")
            
        with gr.Column(scale=1):
            output_highlight = gr.HighlightedText(
                label="Entidades Detectadas",
                combine_adjacent=False
            )
            
    submit_btn.click(
        fn=merge_and_highlight_ner,
        inputs=input_text,
        outputs=output_highlight
    )
    
    gr.Examples(
        examples=[
            ["Yesterday, Sundar Pichai announced new AI features at Google headquarters in California."],
            ["The Microsoft team collaborated with Paris officials to launch the new software update."]
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch()