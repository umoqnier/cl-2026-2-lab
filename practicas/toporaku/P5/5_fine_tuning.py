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

# %% colab={"base_uri": "https://localhost:8080/"} id="7abd4c18" outputId="b1ea2ad6-3be3-4016-e687-824b91b3959b"
# !pip install -q datasets transformers evaluate accelerate huggingface_hub gradio scikit-learn

# %% [markdown] id="822c7343"
# ## 2. Carga del Dataset SMS Spam Collection
#
# El dataset contiene ~5,574 mensajes SMS etiquetados como `ham` (legítimo) o `spam`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 217, "referenced_widgets": ["ec5280a312f546fa89a7a8a45ca225f9", "8db6dafb5554434fb1f5de88727326ab", "d03650f8d44f4e7d9ae6fae32367f910", "b542306ac870481b861935d5882b40b2", "c7d54edfd6bb495b9a47d05ce345a000", "837d06a1ec79406ebaecb04f0e56659b", "0e5dba8b9816412b93d3d044d513c954", "fe6538adf0454a86841953c989d4ee0c", "38ec35fc9ec24b0a87e5b75fbe042576", "28d7c8f416344ce285ae2558a39407eb", "07c5b695d9bf47c38d13cc33f79b066a", "c81ad4f4d7f24b0dbd6ab7c6b107bbd8", "b049988ad5bf4048bda67ff5dc9b306a", "fa4199b7801c4d77ac2aec0de05f1075", "04e0522a3cd34e98b1cb5e171cc69829", "c24daad209f04700a654e3409cee6129", "fd61d94442034155a4eb086c1c8c440d", "a122fe26039b44f0873acde9c9af3076", "859fe2237c1c4f77b361ebd0b79b1456", "ea380a8fa87c4a0aa7ea20a9a97c476f", "25eeaf061c7b4177817023d2882a13df", "d38d17de2aed4457b254fb1476717b45", "45a264e8a5bd43baa79606c8e7fd2745", "42021e51d7784a37b7028a01dc363227", "16fe94ad6d564ded8a9f9d193e3261e3", "d6f9f0eb558d4e32bc3a829f955e70e1", "c01ad47de7404d3ab41181d9a961039f", "70c335d7d8a44a179b8bd2db4a57639d", "a9e1b16c9e8e4bb1988183ef412c137e", "fb8926700ff34fb9a09418c5ac29a192", "20b94920bf94457a9263ed3d9dcedc09", "3f953f08d7c244509f2fc1ec692539b7", "e3dec477fd3a4c1fb27b197887d30d33"]} id="182992fd" outputId="2d97c552-33d5-413c-e05a-ca1779badc3e"
from datasets import load_dataset

# Using the standard dataset identifier
raw_dataset = load_dataset("ucirvine/sms_spam")
print(raw_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="c9157e0e" outputId="844d7de2-24e6-4839-eb6b-61b4347b5176"
# Explorar un ejemplo
print(raw_dataset["train"][0])
print(f"\nCaracterísticas: {raw_dataset['train'].features}")

# %% [markdown] id="026d7921"
# ### 2.1 División del dataset
#
# El dataset solo tiene una partición `train`. Lo dividimos en train/validation/test (80/10/10).

# %% colab={"base_uri": "https://localhost:8080/"} id="a2ba9f7a" outputId="866bb840-90b6-4d93-daa2-c9516de233eb"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 214, "referenced_widgets": ["c6d5b0d024934c05b1bf14c2fd80afbe", "0dd4b2322f204c8292f15756e7ab1251", "baf958e7e30a4403b3dd86ff7e1890f3", "e0fc9b3873ba42098aab265ba252a33d", "a6f4167b1ec64aba955e85680e098229", "b7e13f9903b245c3a34e99c86ea75e30", "0d7caf7e701944f1ad85d8816d0f4d35", "3fe27aa924b145038ae760cfffab6f09", "e6f98838d9f740bf9787f4b8e1701fd5", "0532658a9ab94e919802d598bc15d5a5", "51b95ec7a46543c0915719d46bdd6550", "1c91f051cc9c44cbb245b78ae7af5e92", "c2e788f2612f4755b3edd0b49729b7e8", "e3d5a02b034a478a920efe2bb5620b87", "7827bf348a184ff9b1d7eadeb2f5607a", "9ed8e8bb5aad4ad59e773e5d3d28280c", "d17c787bc8f146459d3ebfeb5b32d0d7", "5a53ec42e7a34db4a70162d5c92a844f", "bd68fc41b5c6482192e4e504dc5b6ec1", "37f33bb1c2224a52944d432c1a2dc3d8", "58653b7a7692425d9795308f086d4678", "6e2f835e5f5c4d5a8d766bc2cd684945", "9cb6aea800ae466e99a6095a475bed5e", "da76c55a4b46423fb7bf375c2ad36aea", "fdcae1d3790b442298f7f83267b11b0e", "57af9dc9cea04f2ba53e1fd67055c16c", "b10dd3151901478f9aab220404ffc03a", "473d61251fe94dbaac049ad950235dad", "3fb1e31a42894ba1b4e3feb74b83b5fb", "5ee8ed3d2a5c46d297715aaf1b283c65", "479d8bfa23bb4dfd93cb5a7cd440b0b9", "0a17a876837b4e70ae2ae98aa98093fc", "292986e2e18c42f380366460d76d055d", "ec492fa832934020b09bf93224ce6d5d", "ac56aa70d63546a78bef4c7a96e54015", "23656cfe9fb142699a5bfc217f406435", "724e0ee4bde24282999c61fe90f515dd", "8f84c896df62419ba4d081dca1aacf02", "622622a11fc94dc682caa4ec1826c0fa", "64ecae11ef014206bc66a1e7503ee0ab", "cadd4b42bc9746d899395b04b535c724", "2c2dbdb0ce884cfc866ec36246e1a1dd", "2f28294e8e184fa089accafea7d1ad93", "7efdae7ec46d4e9a9925061aff3169ae"]} id="c4ad1fd1" outputId="11ce853d-f12e-41f5-ba75-7837aa137877"
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ejemplo de tokenización
example = split_dataset["train"][0]
tokens = tokenizer(example["sms"], truncation=True, max_length=512)
print(f"Texto: {example['sms'][:80]}...")
print(f"Tokens (primeros 10): {tokens['input_ids'][:10]}")
print(f"Total tokens: {len(tokens['input_ids'])}")


# %% colab={"base_uri": "https://localhost:8080/", "height": 356, "referenced_widgets": ["986f27ea9b194d11b014733705bacffb", "8deb0b428422441896b7595199a8b2b1", "91982bc22d0f4097a24b5c23ba9d52a8", "8a99aaa66ed44cfb828f40646d37f231", "5c14f1b93e6f4387af41071c86d1b186", "1d6485a0f4d04dcca90c2829c3919010", "aab187a7082d409fac6d1e4f0bbc6681", "fb4e701f597e44f28e641de5a5549b51", "6232f223e4bd42d9b1b46b013763c3d9", "443a47e62aa24ee08d6290d7fe9b284d", "5bd65d3c29b640009540cbc766830b95", "54750ca95a584fb8a292b8774d5bea74", "d5b2d12cb24c42eca0f94108b47a1d23", "d5a3c789e3a94c699bfb615377f4a3c7", "f5fa7cda156a4295ac42adbd14afe5cc", "92dd9e887ef7464693e1d2e1e3d6b926", "2c8014befa944951b32813c9ec54eb4c", "2f8b60ebf68a4406837b21cc2243917f", "31d90064db214007918854a15d714fa3", "5d0cdd699d484d07897484c739f36514", "7a3a7ff8307c47498d3fc471e275bf46", "353d51d2d0b440108a681af3a28244e0", "75fac28f5fe54b848bf5f99f7470d6ca", "042383ad3354416eb8726bf110393c7f", "703df2076bfe46e09c3da61c90424433", "943f580e6fed4ff0a8bfa3ae5976d2af", "065542739cff485eac37166a2d3a7848", "d58ffc7e5555457697ab1e350438fbe6", "ca0bffb4a7244f08b8dd18935b7d7aa9", "a4e54b3b847f403795cc50eeaaeec0de", "41874e9d862b4adfa51642cd3bac48d3", "72ff7c28877c4908b65b12306752b772", "40a3529bd33841038b614778a0cf8603"]} id="d2ad13ab" outputId="ea0226ea-4f2e-49ab-9cee-5897d9037fec"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["ba46686430914cd9b9c9d7baef3a201d", "4ee70b01854d46928df8edecca6a4811", "d0caffa836c44dfe8fdb99f4fa7c4bb3", "19f30fb1277b433b9f7094f4dcaa14b8", "0b0dee3544ab44429d46e49b3e89c7cb", "e2ba83d2a27d484bb2593e13d3abf0e5", "f12e5eab333d44739f76d4d21d3e3bff", "47f049216ac64fe6866f6bcde9f03657", "6d8af1f630fe4f06aaf27c275eaa3283", "d2d5ff2541f24e6c8504749670a42e97", "e32f30a6d8f34ba2b7eb86634ab58567", "b32d02ca426a493a867ffb6cc5f859bc", "4967b7508ad24351bc56fe505fdce01f", "309c687ff6cf4630a6fda958590ab7fc", "c740823a565c425c898cc38983f7593e", "549c746a1cf94bb69671f02b7a595e35", "ee0b5ccc06db4b5ca50591c318624d9a", "cdeba5cdeff7411ab0b765fbe3703bcf", "2b15f106ccab4cb5a7e0f2acef50c688", "988cfd0ba68b4aa68a0d00e77f9fb9d6", "0f26aaac1fcf4c978cdeacb7b3576c4e", "90c28ad8e42a4cac8f4a443ccd497a92", "74b6571a3a174c37aaa0c1f0e9bf662f", "b5cb666d58654bd6909fb83759e86c64", "9525073a988a4d96a6962a743e0ec975", "e2444e0dfe6d4ab6ae32ce3a8bd576ef", "e918a3c9754c4f9b86890d1dc6c4a671", "f66be6ed0ba843b285bce44703844c1f", "340a361dca07408cb7f1dce04471355b", "23443c661d984db5b393ace514c5f998", "1b0f0521e15d474bbd7c88c9b5460fc3", "d347a06636464cdda178373338a0c913", "1b5fc4113c0343168b594ba20e02577d", "9fbf0a104a114ac384ab9a2dfa389503", "fab8dd1355c44d8599d4020787eafccc", "6b9353e9a9f24e2da92088e585a21b3b", "f97b14857acb4dda9ccc9106350dee76", "b929c5c5ffb843f8ad9ceb6c9085e40f", "2e0d04c110a84d0c957bc18c162e5d9d", "aee68f73ea504330b1d9162ec770b792", "e566cfe0c82d43b6808a3d01117f0369", "41080934c1714c298ddc8b1cbadc13f3", "653fa1f8f2ff467281d1f1a05711b998", "dfd424d3a5924abe9a06566d87bf77c4"]} id="097cf331" outputId="96940bfe-e46b-496c-a6d7-1e0598d29052"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 359, "referenced_widgets": ["cd51041a0b324a12a28495ee1e1ac0a9", "f5cc8b6f08b14eef8aa8c61a1a6cd882", "df901e50039d44029292093423326d09", "f8e81a7f3a6f4463880fbccbf075afe1", "634ddce02068444888fb56ea536ac160", "01bb00f9a9a044af9161e4cb9ed77d5b", "cb4127d1c3234195a34532e260fe7587", "3dd41349e4334dbeb3a990c1783bab96", "1fdd309d23a34b6b91a3c1d2a69db382", "a5d4ba142bd8469682fec4acf3940e41", "163774ed9d054b9dbb515dab8edd562b", "89e6c0d1a5d840dea682833cd39f3a98", "5d92187592124df1a39207572c84152a", "90b58061c07e45b4a4dfd3f7c7fc2860", "5f4439a5b28f4a7ba5ac066bbc67a34f", "c1b245cf8e194ca3910ff70b41e3c53e", "fa654e4a96a14f2aac48f3b0c142e04a", "8a1b46dae5b341aa923efedf039d78c1", "4bbd6812ae084a17999e7ee6c17381a7", "67b5e16f3d3040a9ad0e2925c6a29a7c", "27dd4ae6903645df943fa1ecdb95b327", "39bc156753204cde91d38e343740581b"]} id="9ae4a3e9" outputId="ae93d683-7d70-4521-8e8f-acc6b7fa7365"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 335, "referenced_widgets": ["6cad63eb65814baaa7bc68fb1b4823d1", "82821e6a9bfc4320abf6af9e91a66961", "4bf354e115f64ae0a49aa510f18ba107", "319c2615f4c34575a778e34083f88c39", "6a267e4b0ff84a78966e3ff46bd2cf24", "7e75ef1540ac494387f668ef3668e814", "cf20d5b5775046a786ef8a8ec337b6f9", "83771d1a1c634a0d87f4f5203116ab90", "e06719741f3841a8ac3d8fc794555a46", "b32a35d7af5e489b9618f04a5edbc8a6", "a8248a3aa2ea43d1b496de427caa2d0f", "d8a06c74487240239e765fd56b261bd9", "18c7262937ab4708af16aed88775fc3a", "531c525fa1ab4c9ba7735c012954df02", "f029c60d028f42eda09cc454235982b4", "5539f32d074c4ca893beed0ac193dc8c", "875cbf8997984aa19b5a09ba4af88af2", "7b06347d7a9f40db94a9721498e1fff0", "b268e0c6407b4832b8eb00267f2ecc54", "f8dac904f14841308e2b77ab727a6821", "1ab955945be5433c9ca8af7451878770", "50c1f5ab6d6944b2a9bb6d00db2c06c8", "310d49371ae34abfa39bb7b718e6c2fa", "40e512f6126c4f8db344924bffaf94be", "90db94eda65e42c580b7e609bbd4e86e", "317e4a3b5302458792e6386353a0d6ff", "446bc236679644fcb87d4aefb6397978", "c87f2ea8013541d896d4adfe0682e248", "58a50df2fbbb4cf7a7ffc68fb115121c", "c5ac1573c8a8436caedbada3977334da", "c173deddd3194f4495f1163f42059702", "0bb9825f9aa14273a1a843219956c5c4", "a8935c17a29340f6ae17badcdae41263"]} id="67d8464b" outputId="8d30b693-e0c2-4429-d363-87fbdd1f6895"
trainer.train()

# %% [markdown] id="b0b0e3da"
# ## 8. Evaluación en el Conjunto de Test
#
# Evaluamos el modelo final en los datos de test (no vistos durante entrenamiento ni validación).

# %% colab={"base_uri": "https://localhost:8080/", "height": 193} id="db06535f" outputId="a5f8f22f-23b2-491a-e724-5918b5ded506"
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Resultados en Test:")
for k, v in test_results.items():
    if k.startswith("eval_"):
        print(f"  {k.replace('eval_', '')}: {v:.4f}" if isinstance(v, float) else f"  {k.replace('eval_', '')}: {v}")

# %% [markdown] id="e6446375"
# ## 9. Prueba de Inferencia Local
#
# Probamos el modelo con algunos mensajes de ejemplo.

# %% colab={"base_uri": "https://localhost:8080/", "height": 171, "referenced_widgets": ["9ec5f58af1ae4a38b9b62017fa3ba2c6", "e963fc90355544d59fefa0f5a0a0fbdd", "f084e909127b41888bfc0af84fcbc565", "dffab843c8254202856b50760f2b7970", "3507965bcea24b73afbf63c5d1551d0b", "45773de35d304058b784b984f485fca0", "fe66079113c94954aadeb432c764e6a6", "1b6a6a6616164ac08ff7790287c961b5", "f27cd7bbb99e4ca786850f38ee97f990", "3e8ef2943e5144898ac544444577ef65", "1d01aeb7d26241ccab72a7160bf86f56"]} id="bfe89bee" outputId="e89bb0f8-6f42-416b-bafc-bc0c27d358cf"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 229, "referenced_widgets": ["eed50d8a291b453aac59d6a6e92ac3a5", "f3d2c0dab1bf4b13b7944d4183e778d1", "e43624808de442b5b2196976b5ad4c68", "4ad719f4ed684e4b9fea8d8e6e5d9897", "6ccd65a412844eba914236aa43b62375", "db8f324100b84f5dadac31f826c951f2", "006ced28f9c440e996941a9a9b4576ef", "860f38a3d37b445cacdd196b24552017", "664ab6156a344cef9d91698c7c7b1a31", "6a23963d1ead4b3a8946c8c4295ca108", "03b4208a8aba42ddae459823f2b530ff", "1b274fc52beb4d63b3fd724b49e71f35", "c8d73dc7f99e489abbd27bfd47b0e93c", "b3cf4609272546c5945ba34f39bcf882", "a34eefb20ca3408e89cad9c617be0337", "1185dffdfb3f4fd28e3519afc2e8798e", "1e32ce67f8564569b77d0dad1c6006ae", "280070ae82e240879564a157eb8d29dc", "a543497e9627404aa331187e561d80b8", "d295ef5a57144553addc40f20ff0abfa", "2e0c3ef83ba14bd8b0d773339a10842f", "56949dcc83f24dbf96439f98de29d110", "be0fb2d9a0aa4bd6be13dabe1b3d01b0", "8034721cee5f478290370762ed1cb9c6", "44f87e9b8feb4eb8b938cab9b9708385", "b15f89fc3f814b00987470d558cb9d1d", "09e632f52a0c4bb98606250ce8750033", "26d6b3cb4d064f759076c0952504fb4c", "2b2c6f9f07584ba18ab010410a66e4ee", "bc503e96a22a4fc48cedb310590b1760", "099bdb996ebb452a8a124a7d6414f790", "73d012e2b5824968b4208be4554d6c62", "e21b7060106045fcbedf6b60770755a6", "916a0e78dc01458da3ee1c8f9a587bab", "16b7329254e442c7aba24f51d88d1e6d", "aa160381706842658d9f9f146471b33a", "e461b757d24042fd85cbc7ec2ce029da", "494a941adece4d72a7b2e3924fc1f509", "2743c631e864413695914444321d573b", "8d1602ca85824e33b3c6972b202a96f4", "e92162ee52a14f94985902a0c1ccea7a", "5d2d4f7ac113436d9d52e8ea542de8cf", "8108220d707449e0a8bbf99604cf941f", "8db9f43434244b2f97a9289d861721b0", "5837ebf85b364c529252906b2012b7d4", "b7f5e8af7467463e98c860e6213fe685", "0782e149b51c4e3698b37a401ec44d19", "f8c76562ef1940369c88535c012dc4d9", "bf1c70cef50144bcbda6b5388ad2ab23", "166e88d1cc1f4b5589459f0c5194ffb6", "0940405885594d5685cd84c227b0a0f8", "b96f049fb03d47a694d657aa2161b488", "1c63dec282f34285ab508590da3d1ffa", "f8a372f595304660bb4aa94d28098e3b", "0113b2563a3c4c65811eba443ab676e6"]} id="260bff3a" outputId="85d4cc1a-ba96-48a2-af76-61ad4dbc50dd"
trainer.push_to_hub(
    tags=["text-classification", "spam-detection"],
    commit_message="Fine-tuned DistilBERT spam detector",
)

# %% [markdown] id="a18d4cde"
# ## 11. Demo con Gradio
#
# Lanzamos una interfaz interactiva para probar el detector de spam.
# Con `share=True` se genera una URL pública temporal.

# %% colab={"base_uri": "https://localhost:8080/", "height": 623, "referenced_widgets": ["9d25e7f31e734d0da03426ef9897b1a5", "8987cc8f49724025bfbaae7b916f4880", "e9bfc2ea88bf498fa7329176fd6402cc", "4ed17ecf814d423b8fa7f141fcb7bde8", "0de2ab4afe2d44a784129f2b9596a38d", "9f7bc03025c44d84831da7f7043ccb7f", "3ec4f13a6a9d4ea690f12a127e80ba67", "11388a31ea7d411e916dd3b93c23b905", "89d0e44d815843d1bc72aeab0f235705", "b5d0824ee8654ed39326ccbcd8e40c20", "d38542716a2c460a88f8ccd6da80d0aa"]} id="5e62e990" outputId="5297c05b-c7ae-4334-ef00-fa19fb488755"
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
