# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="d984e947"
# # Práctica 5: Fine-tuning de un Detector de Spam con DistilBERT
#
# En este cuaderno implementamos el ajuste fino (*fine-tuning*) del modelo `distilbert-base-uncased`
# para la tarea de **clasificación de texto** (detección de spam en mensajes SMS).
#
# Utilizamos el dataset [SMS Spam Collection](https://huggingface.co/datasets/sms_spam) (~5.5k mensajes)
# y seguimos la metodología del curso de Hugging Face y la clase de Lingüística Computacional.
#
# Al final, el modelo entrenado se sube al Hub de Hugging Face y se despliega con una interfaz Gradio.

# %% [markdown] id="441edfdb"
# ## 1. Instalación de dependencias
#
# Ejecuta esta celda si estás en Google Colab o Kaggle.

# %% colab={"base_uri": "https://localhost:8080/"} id="7abd4c18" outputId="2ad97f31-a6c7-4594-d9e8-c94a7dc42626"
# !pip install -q datasets transformers evaluate accelerate huggingface_hub gradio scikit-learn codecarbon

# %% [markdown] id="822c7343"
# ## 2. Carga del Dataset SMS Spam Collection
#
# El dataset contiene ~5,574 mensajes SMS etiquetados como `ham` (legítimo) o `spam`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 217, "referenced_widgets": ["32b8365213154ded8c87ce50daca2f55", "94e25e73d3d64f309c6ad87cc1755518", "e4af345b5ac546ad900afb94c980e35d", "3967f11263c446e08ce1b895ca1caa8d", "6bfd0788d5a14152bcb1acc243791f8d", "dc12ca4abcbb409ba1f9a5f0ab5189f2", "c2288336d43b4b2096be4fd2da5ee8bf", "db837d72f16441f08aa0da149f2d11fb", "3f7f63dd4ff44758b96f8e969c674937", "410578ff2cf84e9b9a1c8427a100549d", "800361caa12740cc94b68661078f3f5b", "1f0e684457a94cd98ff821d42757302a", "30f8a18ad3ac45d0b61742347fd2f32f", "6e890fbc322040a1a1bf1b921a34b362", "541f9ce9051d41a590f17399b712fb21", "8ef51edd943743a489bcb1ba043b204a", "720a20a1e6e6425aabf289fa1ae7dd3d", "bfe7af40319e4b3b972e2e9f0b5e6576", "6ce892a015454fe787da433d26aba504", "82d6235e38534de9b86cc4bbdb61f578", "d0883a5b5fc24c548efcf363465c972d", "f9bb962670cf4f21aa53c13891e10ca8", "fff3d854d139435ca78e972ef68f9ecd", "6dfa77c476b140c19acc1b0859317ad1", "8c06811c8f8c4f31a774ab2372ded999", "799d7fbb8ba64c40ac230aee4ac134f9", "e54f9e60c397475fbe436f488a9bdb25", "15b08458a2604b63b58dfcd3954c2743", "07e968ceb2654bbba987fc5c87010321", "7aefc8198aeb489795c223a9d460b452", "0ef837b997f04c7b8f82576d83b1b82c", "f1d69eb01e02435daad8e0248a500b57", "a766ae23e5be4789912614aae8030c3b"]} id="182992fd" outputId="33839f45-d243-41b4-9ff9-f60601617c99"
from datasets import load_dataset

# Using the standard dataset identifier
raw_dataset = load_dataset("ucirvine/sms_spam")
print(raw_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="c9157e0e" outputId="e401dddb-fc75-4bee-8d6c-206ce81797ed"
# Explorar un ejemplo
print(raw_dataset["train"][0])
print(f"\nCaracterísticas: {raw_dataset['train'].features}")

# %% [markdown] id="026d7921"
# ### 2.1 División del dataset
#
# El dataset solo tiene una partición `train`. Lo dividimos en train/validation/test (80/10/10).

# %% colab={"base_uri": "https://localhost:8080/"} id="a2ba9f7a" outputId="db1a9ec1-169d-44e4-8543-1e931d2848b4"
# Primera división: 80% train, 20% temp
train_temp = raw_dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

# Segunda división: dividir temp en 50/50 para val y test (10% cada uno del total)
val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

from datasets import DatasetDict

split_dataset = DatasetDict({
    "train": train_temp["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
})

print(split_dataset)

# Distribución de clases
for split_name in split_dataset:
    labels = split_dataset[split_name]["label"]
    n_spam = sum(labels)
    n_ham = len(labels) - n_spam
    print(f"  {split_name}: {len(labels)} total | ham={n_ham} | spam={n_spam}")

# %% [markdown] id="285f27a2"
# ## 3. Tokenización
#
# Usamos el tokenizador de `distilbert-base-uncased` con truncamiento a 512 tokens.

# %% colab={"base_uri": "https://localhost:8080/", "height": 214, "referenced_widgets": ["bc849ba748b941b1bd0b4983cd8a08c4", "84683f61fb3340aa899d960c9d08c65e", "21f91f2ecf2c4841bf92a7282e1f75a2", "6d069851df2948b293be5a5e0d387c4a", "d131424003aa4b259524f4a2071bfdb4", "25613a78ba404026a26015979edd3d07", "33e6b4e588d141c49386466423230423", "62eb075208194786a4986af44a199f80", "f44c0fe3b07a4231ab0fbd30b4d5e808", "5b02918f21f34e0b9b65eec96824a5b5", "78925bbc865f44e28adb7da459fb863c", "f7c65f4a28204bd1b8820dbba1bb24c1", "d4b57e8edd22424a88afb3a32c3f6760", "1dd70655b67f4c0dad760e4ea4d3e479", "c2b6dbd6916a4d04ac82c2e21fe71560", "f929d30aed2a485aaf26667ffc3eba32", "d20c2245258d428ba9b96bd939212b65", "2650590a2a6346838b39a40b91565ceb", "4e93fef0f1ee45248eb0b96423f32a6e", "7b2476541b3441399d9d91b7c53c6c3d", "e34a80f6881c4d029e7759b54dc7e1b9", "2ca81529ca534984a98170359b36bf1a", "45e87e256ff545afb6f7f819f2ce18eb", "c9f210a08bbf495da020af20fbfbb8db", "d9c4a89a6fe64027ae93e2415bfeaaf2", "279cf86827eb49f9bfcf49e129e445fa", "3b8afc41e8884b11935aa9f703aec4ff", "19e48d1149fa4e0bb1962c96e0831855", "3f0ade7192b846bd8c7b21ef38d8b58b", "eb1de437decd4cb189578a5389b73527", "c54594039da5498cb4778171d435b24b", "d4b062adbaa04cd2ba40e59bf995fb37", "b57faf18fb34404ab17a0e5d516c40a6", "5b1ca7c4e5af45ee81434c2e65f8acbd", "6690381fab854729bd8abd9677b1b71d", "71ee135b9d644efb9503b499bcaf7853", "b4ff46715e634d91b0eb88fc64dcbd47", "1bfdf3dc51414ebfb765459ad06e9164", "89175ba3a8464f73a3e660d37c50d29c", "8e763518aa0543a2849faaa31378ca01", "284a729f33bc496c8971d1d5609e15a3", "5963899760c3407bac7eceeec0ae3dad", "ab95f5bbb1454350a0db57caff90e53b", "5654e1d571ad4b0099e7a5999fe89ce8"]} id="c4ad1fd1" outputId="88035e71-447b-47fa-f028-07742f075784"
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ejemplo de tokenización
example = split_dataset["train"][0]
tokens = tokenizer(example["sms"], truncation=True, max_length=512)
print(f"Texto: {example['sms'][:80]}...")
print(f"Tokens (primeros 10): {tokens['input_ids'][:10]}")
print(f"Total tokens: {len(tokens['input_ids'])}")


# %% colab={"base_uri": "https://localhost:8080/", "height": 356, "referenced_widgets": ["bde89d4ffe3448e3be67eeb3c7d99479", "c984c71c205447bc85089d60b8ae79da", "7c31ebbe92f3403687d92d6565b1b125", "2ef24d9743df41ce86536dd7a4138f6c", "4ed058c02a97466bacb759b5e9a25cc1", "d54f971bdfba4bf4b56402c8d5639820", "840779def1244e57aaefd5d4dd4bbc66", "f5614339ec1a4660bf3a53f3bbc41094", "1831ed64851442d6a6656448e3baa606", "e4b24e2bd5cf4b8ea082b83e8d5dc80b", "fac4877df95c40cc84e48755230ed2ae", "68518c76daff4611ad18510477a52723", "fb37331a995c47c68bd40a7f41fdf773", "af289a14be864756a6de5cfae3de91cd", "6a3e34611a584d039c84fbe1a8862469", "7420338f69944d1cb966838bd7b4dc8e", "e84ef8126c914a0ab4e29d57eebb36bc", "946faaa099944bf89483358e9868b6f9", "ad10e86b0f8a455582361f042f6d913c", "0506a8f0779040519c66f87c39a0c07c", "2c681d6d882e474ea08f6afd18ad0043", "16b9c63936f845fdb9633c3256bc49b5", "370d8b6c5ee546d399a971374b8ebda9", "93b10ae9f58549db8682854137f16a2e", "798003784a3347389a01ed54f2aa4763", "6793675bdb2244afba71728fbc3e14db", "aa0481dad1354f1ab6885e4c640bf83f", "5fe25780dfbf4a6cbe7aceb3f531de30", "37a76dc960204f7a9a0ae85f8fa93bb7", "7f0b7d6cf40b4e258ac9d9b19ce6175d", "10f932cf5e904926be9ead8ec48f5e83", "010f6cded23a41e3979d0c1fb0408537", "375a2637c8ba423c82d63c9c9e99e3a2"]} id="d2ad13ab" outputId="d61af830-eaf5-48c7-ab33-383cdf258ba0"
def tokenize_function(examples):
    return tokenizer(examples["sms"], truncation=True, max_length=512)


tokenized_dataset = split_dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)

# %% [markdown] id="087aaee4"
# ### 3.1 Data Collator con Padding Dinámico
#
# Usamos `DataCollatorWithPadding` para aplicar padding dinámico por batch,
# lo cual es más eficiente que padding global.

# %% id="b968417b"
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [markdown] id="76930dec"
# ## 4. Métricas de Evaluación
#
# Definimos una función que calcula **accuracy**, **F1**, **precision** y **recall**.

# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["4acbed382df74064bddb369512c51918", "c7c04f5a912f4d85a1008c29fa2070a6", "2a47fce9d818495eb4109315992ee08f", "49499fa1fe554cd9a66ec385bf021be6", "a92d2de0857344939a01b2d970bae168", "dc30a3cce9ad4030b9326155962a9334", "7e1b4c8cab0041c7904caa60265a8cfc", "313e6e3c66484e72bfbad6ba5959dc42", "ec2f80b5cb30453ab9199afde66c8134", "f75924af83ea4187b6e413ac2012ad89", "06c4c9aaad464118a840494d2e3723b6", "e6a5b1fa6db8421193217a0cf45ea66f", "525ce0163d2042a2a93b64425f88824b", "442a557315454e66bbfd8353758247d5", "94192f6eac5f421e9239c7515d7e56ba", "ed2b138f2aaa4b6fbb9d5c8f1350d1a0", "0434ca78646242b0b018f97c854b114a", "174490a58d2b496d83f259040f8e7548", "d73c72567061447eaf5a19da3fd38c28", "3394f70bad4d495a830e5b03c98cbbcb", "cb0c986066d7476b8d0d1f540689ebc6", "aee3bc4fa8324ac1a4e91317d0b247dc", "2e7d3874603d4bcf9608b4337a0df50c", "3b2648a4d97d445a983b8689cb8c99fe", "1b3c56cd5b1940a886a29bc142be3e8a", "84c08b41dd2d4772b78e3eeadfa03123", "2f208a9e4a1f446ab1e0b3eaa0a5871c", "bbb5823bb4b04d4e96f7b784e524fb24", "ddf849ab839445d2800e19414c1bac1a", "9c2e5c8bc0b14d9c9096113b6641c9af", "2dd611ddfd5f4c54a1b1e3ad6692e7b4", "86baf9ecac934c50b6c787dbe428afc0", "c2f34953cb564682b549cee46ab97bf7", "d6d32193a556433c909121011da5cd17", "ce8bcad5e7be493a82721a39d8a3f603", "96987e32a198469889f6c7d9f7bf6246", "3a5074314fdf4a7289f5e4070cbb5352", "373e0e411fe64804aeb7ad680f746918", "d7236212db3e4b678b64251f15af1264", "4d5d172e1fe04bffa4b93ad00f2bca11", "8476a7aa4db24eb8846916b830d8dbff", "1cf5db104e354c418a5372317f4cad90", "722cbd15aac34fb89ad319075156f1e9", "a83410e003b74cd2b3d49d312b08f699"]} id="097cf331" outputId="0d3a60d9-abf0-4952-da1a-a94618d0a221"
import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    prec = precision_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels)

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "precision": prec["precision"],
        "recall": rec["recall"],
    }


# %% [markdown] id="c2042a89"
# ## 5. Autenticación en Hugging Face
#
# Ingresa tu token de escritura (*Write* token) para poder subir el modelo al Hub.

# %% id="80dce668"
from huggingface_hub import notebook_login

notebook_login()

# %% [markdown] id="647485ae"
# ## 6. Carga del Modelo y Configuración del Entrenamiento
#
# Cargamos `distilbert-base-uncased` con 2 etiquetas (ham/spam).

# %% colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["13c39a85a03b48a78e79988d9b9c2eee", "ffceabf2d82e4b1998d0a8d4277c2e07", "2c0f3d7b470948adab598fb4460cd42a", "05a8c016b8724501b24e6e77a7e1d1be", "7a6c4a2baf034280a5076a0d76065253", "7a8f6d0da6334898acdb348226c98320", "55f791304e2d4827a8cee8221fd88e01", "3875e04eee8e4bc9ad4f8ea27be9aac8", "765ce8807ff64d6bb14d5d2870ba0f30", "a9dc79ae8ca843f2a7ebdb86c578ad49", "ba26b04e91694aeb9e7fd101fe12c510", "8b081be164d44b0493ec083d892da0e6", "bea0d63427fe437094120a61dc1351b6", "3212ded0523c4f0ca5d704c46835e735", "448df217387343fa93701b2c0ba12ac5", "3652600346a7460eb8c9629549dc419e", "0150ebd4fe8d48dbb7a076f01fe320b7", "6836eccbca87487a908e922c8d028f32", "aea50ae957384f79908c56b9be8ce3d2", "4a7f17ee307d4207b458075e81f3713b", "efaec61770d64f69a60b9d0ba8f909dd", "8b3aa0741cdb496cabeb7b5bb7e163dc"]} id="9ae4a3e9" outputId="ccfbdc14-ab5e-49c4-b016-d517a6bc92b3"
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label={0: "ham", 1: "spam"},
    label2id={"ham": 0, "spam": 1},
)

# %% id="15e32d0d"
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-spam-detector",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    push_to_hub=True,
    report_to="none",
)

# %% id="a8ff1cac"
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown] id="3fb9db6b"
# ## 7. Entrenamiento

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["04373af686c44b4495904e1bc661fbc1", "3f60bb154e9f41e68f42ef9a6f4c4356", "3bf5f4e4816b4f6f89c0875284f02f83", "b775dd00b029487bb2ef275c48371325", "9f086cefb0b14a70bdf31effa18d8511", "e213e2ba03294a5b9e8e6183ab7501a3", "2fa3d68c2f5141d6aad436042468f46e", "ba35e903c3b846dda244077ddd2db9f0", "bc35a9a5b15748a5b67ab473e4dfef7f", "ee0565e41ea54dadb61256d7b4f9bff7", "cf4d6a44f0a44674b3b34836e55d1083", "b1c1e9b15daa4568b385e9323d18d5c8", "6946e74c829344f083bf7043f2bc24d1", "41ffb57240f543e8affc565c84ae73c8", "4bed8869ef054cf28efa9357043d17b2", "40fe3a1e0bde47d1b67e3491d766d89c", "b6204466d2c04dd799688ca7a061d4a8", "160689ec7fbe43fea7f53d0b89d8c83c", "7a9bb1453b6e4353ada0be10838d3e6b", "e6c76ca5a9a64850938914941467daf2", "56d06a7b7e954cd6b2ea4cd85359ae26", "6a877123c0e544d5bec9b8ba5062ff94", "ae150c1ef6ed41e29899bb0595b8006c", "99b077f0d27543acb6767b2fee659624", "00ea4a80d5c948efbaefbdd72e53c7cc", "8f7ec783277943739262164dbf9bd3e1", "d9a5dc2c35ae49659784d1b63da29651", "24cc36ab6b294e30b50a2f4f7422e0bb", "ebec78ee8a8e49ec88c0ca0152131cde", "7a1aa079de334af6bcfa6c2cd884ae7e", "bd814a994c5d49a5ba85db5bcd53ff75", "6e07b6e0d00642a8970829fd2a91bfd8", "44868a712d6847dea2e94cc84386bc41"]} id="67d8464b" outputId="3dc8095e-1535-4e42-b411-12c00b7e63c8"
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
try:
    trainer.train()
finally:
    tracker.stop()

# %% [markdown] id="b0b0e3da"
# ## 8. Evaluación en el Conjunto de Test
#
# Evaluamos el modelo final en los datos de test (no vistos durante entrenamiento ni validación).

# %% colab={"base_uri": "https://localhost:8080/", "height": 193} id="db06535f" outputId="ad894bc3-127a-4b99-bfd4-1dadadffce22"
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Resultados en Test:")
for k, v in test_results.items():
    if k.startswith("eval_"):
        print(f"  {k.replace('eval_', '')}: {v:.4f}" if isinstance(v, float) else f"  {k.replace('eval_', '')}: {v}")

# %% [markdown] id="e6446375"
# ## 9. Prueba de Inferencia Local
#
# Probamos el modelo con algunos mensajes de ejemplo.

# %% colab={"base_uri": "https://localhost:8080/", "height": 171, "referenced_widgets": ["925e3b2cb11a4f38a71a5c4bc3b20439", "765d544fdcb343288512ead98702aaea", "9115708cb644498bae8da9f0d991fb9c", "761d0614948e4dcab338d65a4ed60276", "dfb213b02e634d79a382f9552c63ed63", "d26a25f8e8df48fe9e0465a2961546af", "ad488f599dd549dcaa050234465c9141", "6cb552e42cfe4168a77246b8bde2aa47", "87eab53fcc27456aaaea4bd8bd00de77", "585656931b7e412792742fd5cf23d919", "9019057f7f5d45b080b542bbf88f3b78"]} id="bfe89bee" outputId="002b7675-6d90-417a-a0b0-540ffa3083b8"
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-spam-detector", tokenizer=tokenizer)

test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Call this number immediately.",
    "Can you pick up some milk on your way home?",
    "FREE entry to WIN £1000 cash! Text WIN to 12345 now!",
    "I'll be there in 10 minutes, just parking the car.",
]

print("--- Pruebas de Detección de Spam ---")
for msg in test_messages:
    result = classifier(msg)[0]
    emoji = "🚫" if result["label"] == "spam" else "✅"
    print(f"{emoji} [{result['label'].upper():>4}] ({result['score']:.3f}) {msg[:70]}")

# %% [markdown] id="08d84a55"
# ## 10. Subir Modelo al Hub

# %% colab={"base_uri": "https://localhost:8080/", "height": 212, "referenced_widgets": ["89465a15f2df4fd6b731caa5fb4cdec9", "dbe062ecb891438a90f5b037bb4c8612", "b920c1fb82a74802b43d7f282a2523e8", "3fd5a62eef954a55bd4936b91e4e4c7d", "0cb4855c588a4e2fa2dfd7ab00f49776", "80890425e2b4401f8577c9a344433286", "72b3f98c6e7d40288a2a31be9eb08649", "b044f7bd8d1b4965a24e1ea36fa955d4", "1bbf38cf37d449bea51ab75060f20e43", "0557a26fa4c844d891c69d77dfbd5a89", "fca8004159164527884ad9af7789cc75", "e263146fb0514d45af52736b85c94824", "d3ebf3829b964e1d9db44524889e412b", "0c318ce07d6e4cc0804f642b35b93468", "afadf18729f34e9d90e7b138153f1b3f", "e3079da0152a4e3088b0948e8b47a956", "ef6418481b85484a8f7ed07025c282fb", "9ad514c055bf43639bd3e7d244dbce03", "68016dbeb0c445098b649bbf69d043b1", "8b9417a27f784367a29ee9ebdfa00c40", "b2b4defab8eb4cec8ec69481e6fd1311", "86ef51ad24bc4130a57032438e44f04b", "f548b08e578a4c86b16cea242e6552d3", "6c9a28fba69e47b7961e7881c2b7b60a", "4134b74c51a94c1d92d3ed1700a7047b", "1b8907ae56dc409fb2de6563b9d89a4b", "e8be3783b5f74ed0b0007a5792299016", "2324e3d15a5143f79efbd9ac8f7cfef0", "12feda79d989444fa042dfdbfcc3b8b4", "0a26b230191e496bb7b770e5c3aebe44", "934561efbd48413894b180d4a31854c2", "6af077f5c1464a38ad0b5c2fdab035d3", "ba97316be04a413a980e911fed08144a", "8cebd53ee08a4a9fbd9be79d0e37aa71", "d1735ddd580d4c438a97494de3b38a42", "eb11e080553a42ea9ffe3648b1460135", "2a256d79ccde472c9ff8ae5c4c225dd1", "d6828e1d2a2040b893c04ec42a0a6e14", "475e992dfa444f4da13846b4dd627348", "e1f3100e7634415796aed59dafaa1935", "a2a007cd92fa4fe8945ef53a94003c06", "cc060408f8454fe0a9e1f98dc7e860af", "3e47b5a5a47a4ee38a50e27caec07f03", "35087ba345974dbbb98445e17c5e5bfe", "a6b3bfb7fe914c3ab66a948f5a09bcf0", "12abf556867e4e49b478614af2ac1878", "ced8fec27b464049adf52ca8d0a6261a", "5981e0834ce74222afb1ee434c50b2f7", "5f7226a07492413f9561695700d0756f", "4e15063726df47629f6f06da30b5c109", "72be5061318847a893788f6cd6e43f09", "32aad3a0a4434ba4b66ea540fd369928", "76b2707df1b547b7b25a822c1c6315c3", "72ceb3a6b5fa4385b45d07de337e2cfc", "c932992822bc499aaa2e0d4cfc2c26e4"]} id="260bff3a" outputId="67ddba5d-4544-4a5a-afda-7199407660fd"
trainer.push_to_hub(
    tags=["text-classification", "spam-detection"],
    commit_message="Fine-tuned DistilBERT spam detector",
)

# %% [markdown] id="a18d4cde"
# ## 11. Demo con Gradio
#
# Lanzamos una interfaz interactiva para probar el detector de spam.
# Con `share=True` se genera una URL pública temporal.

# %% colab={"base_uri": "https://localhost:8080/", "height": 623, "referenced_widgets": ["1f997d4634dd4dc8a95488739b466a30", "22aedf51425e47b7abae5bffc9dda4df", "5252b750c0eb4e87b064decebda19a9e", "dbf3078617f34ad9b3bc0fe69ca271d2", "548e65cc0f854ca3ad83de54a96c66e6", "b2eea034face46baa9115e59811badee", "b5d9b510786a4b8ba8e2cbc3d1b3b316", "f58d989d490e4aa68c0d23ca08029b0c", "0754aeac888142e0b8b8a7b3c0152799", "a708a917d9e9436baed592462c1522cc", "902fd7413f184dbaa158fb768f68ac75"]} id="5e62e990" outputId="2bcf8d9d-6340-40b0-e7a0-3d0e2026d25c"
import gradio as gr

# Recargar pipeline desde el directorio local
spam_classifier = pipeline("text-classification", model="distilbert-spam-detector", tokenizer=tokenizer)


def detect_spam(text):
    if not text or not text.strip():
        return {}
    result = spam_classifier(text)[0]
    label = result["label"]
    score = result["score"]
    # Devolver ambas probabilidades
    if label == "spam":
        return {"spam 🚫": score, "ham ✅": 1 - score}
    else:
        return {"ham ✅": score, "spam 🚫": 1 - score}


demo = gr.Interface(
    fn=detect_spam,
    inputs=gr.Textbox(
        label="Mensaje SMS",
        placeholder="Escribe o pega un mensaje para analizar...",
        lines=3,
    ),
    outputs=gr.Label(label="Clasificación", num_top_classes=2),
    title="🔍 Detector de Spam en SMS",
    description="Modelo DistilBERT ajustado con el corpus SMS Spam Collection. Escribe un mensaje y descubre si es spam o no.",
    examples=[
        ["Congratulations! You've won a free trip. Reply YES to claim!"],
        ["Hey, can you send me the notes from today's class?"],
        ["WINNER!! You have been selected for a cash prize of £5000!"],
        ["I'll pick you up at 7, see you soon."],
    ],
    theme=gr.themes.Soft(),
)

demo.launch(share=True)
