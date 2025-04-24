import argparse
import anthropic
from transformers import pipeline
import openai, re, random, time, json, replicate, os
from openai import OpenAI
import pickle

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta-llama/Meta-Llama-3.1-8B-Instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"


def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response

def query_model(model_str, prompt, system_prompt, conversation_history=[], tries=30, timeout=20.0, image_requested=False, scene=None,
                max_prompt_len=2 ** 14, clip_prompt=False, pipe=None):
    if model_str not in ["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', 'llama-3-3b-instruct', 'llama-3-1b-instruct', "mixtral-8x7b", "gpt-4o-mini",
                         "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview", 'llama3.2-3B', 'llama3.2-3B-lora', 'llama3.2-3B-lora-rl',
                         "HF_meta-llama/Meta-Llama-3.1-8B-Instruct"] and "HF_" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    for _ in range(tries):
        if clip_prompt: prompt = prompt[:max_prompt_len]
        try:
            if image_requested:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": prompt},
                         {"type": "image_url",
                          "image_url": {
                              "url": "{}".format(scene.image_url),
                          },
                          },
                     ]}, ]
                if model_str == "gpt4v":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt-4o-mini":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt4":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt4o":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]

            if model_str == "gpt4":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub("\s+", " ", answer)

            elif model_str == "gpt4v":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub("\s+", " ", answer)

            elif model_str == "gpt-4o-mini":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                client = OpenAI(
                    api_key="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"
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)

            elif model_str == "o1-preview":
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12",
                    messages=messages,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub("\s+", " ", answer)

            elif model_str == "gpt3.5":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub("\s+", " ", answer)

            elif model_str == "claude3.5sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]

            elif model_str == "gpt4o":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub("\s+", " ", answer)

            elif model_str == 'llama-2-70b-chat':
                output = replicate.run(
                    llama2_url, input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub("\s+", " ", answer)

            elif model_str == 'mixtral-8x7b':
                output = replicate.run(
                    mixtral_url,
                    input={"prompt": prompt,
                           "system_prompt": system_prompt,
                           "max_new_tokens": 75})
                answer = ''.join(output)
                answer = re.sub("\s+", " ", answer)

            elif model_str == 'llama-3-70b-instruct':
                client = OpenAI(base_url="http://localhost:5999/v1", api_key="original")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model="llama33_7b_with_openai_api",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)

            elif model_str == 'llama-3-3b-instruct':
                client = OpenAI(base_url="http://0.0.0.0:4999/v1", api_key="original")
                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                messages.extend(conversation_history)
                if len(prompt) > 0:
                    messages.append({"role": "user", "content": prompt})
                response = client.chat.completions.create(
                    model="llama32_3b",
                    messages=messages,
                    max_tokens = 200,
                    temperature=0.0,
                    extra_body={
                        # "use_beam_search" : True,
                        "n" : 5 ,
                        # "top_k" : 50,
                        # "best_of" : 50,
                        "repetition_penalty" : 1.5,
                        "length_penalty" :1,
                        "early_stopping" : True,
                        },
                )
                answer = response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)

            elif model_str == 'llama-3-1b-instruct':
                client = OpenAI(base_url="http://0.0.0.0:4999/v1", api_key="original")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model="llama32_1b",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response.choices[0].message.content
                answer = re.sub("\s+", " ", answer)

            elif "HF_" in model_str:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if pipe is not None:
                    response = pipe(messages, max_new_tokens=200)  # [0]["generated_text"]
                    answer = response[0]["generated_text"][-1]["content"]
            return answer

        except Exception as e:
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


class ScenarioLoaderEHRNote:
    def __init__(self) -> None:
        self.scenarios = []
        with open("/home/htran/generation/med_preferences/AgentClinic/data/Noteaid_test.json", "r") as f:
            data = json.load(f)
        for key in data['text'].keys():
            note = data['text'][key]
            scena = {"discharge_note": note}
            self.scenarios.append(scena)
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"]
                          for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"]
                          for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None, literacy_level=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.literacy_level = literacy_level
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race",
                       "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """
        ================
        Cognitive biases
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def generate_literacy(self) -> str:

        literacy_levels = {
            "first": "You have basic reading and comprehension skills, similar to a first-grader. Use very simple words and short, clear sentences.",
            "second": "You have early elementary-level literacy skills. Your understanding is limited to simple sentences and common words.",
            "third": "You can read simple paragraphs but may struggle with complex words.",
            "fourth": "You have an elementary school reading level.",
            "fifth": "You can understand simple health information but may struggle with advanced medical terms.",
            "sixth": "You have a middle school literacy level.",
            "seventh": "You have a developing literacy level, similar to a middle schooler..",
            "eighth": "You can understand general health discussions but may need simpler explanations for complex terms.",
            "ninth": "You have a high school freshman literacy level.",
            "tenth": "You have a high school sophomore literacy level.",
            "eleventh": "You have a high school junior literacy level. You understand most common health discussions but may struggle with technical details.",
            "twelfth": "You have a high school senior literacy level.",
            "bachelor": "You have a college-level literacy. You can understand most health discussions but may need clarification on medical specifics.",
            "master": "You have an advanced academic literacy level. You can follow detailed explanations and understand many medical concepts.",
            "phd": "You have a high academic literacy level. Use precise and technical language as appropriate, assuming familiarity with research terms.",
            "default": "Your literacy level is not specified. Adjust explanations based on the complexity of the topic."
        }

        # Normalize input and fetch corresponding literacy description
        edu_key = self.literacy_level.lower()
        literacy_description = literacy_levels.get(edu_key, literacy_levels["default"])
        return literacy_description

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend,
                             "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the educator's response: " + question + "Now please continue your dialogue\nPatient: ",
                             self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient who has recently been discharged from a clinic visit. You are having a conversation with your educator to better understand the information in your discharge note. Your responses should be in the form of dialogue, consisting of 1-3 sentences."""
        symptoms = "\n\nBelow is your discharge note: {}\n\n".format(self.discharge_note)
        literacy_prompt = self.generate_literacy()
        # import ipdb; ipdb.set_trace()
        # return base + "\n" + literacy_prompt
        return base + bias_prompt #+ symptoms

    def reset(self) -> None:
        self.agent_hist = ""
        self.discharge_note = self.scenario["discharge_note"]

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race",
                       "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]
        self.conversation_history = []

    def generate_bias(self) -> str:
        """
        ================
        Cognitive biases
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""


    def inference_doctor(self, question, pipe, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        if len(question) > 0:
            self.conversation_history.append(
                {"role": "user", "content": question}
            )
        # answer = query_model(self.backend,
        #                      "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ",
        #                      self.system_prompt(), image_requested=image_requested, scene=self.scenario, pipe=pipe)

        answer = query_model(self.backend, "", self.system_prompt(), conversation_history=self.conversation_history, image_requested=image_requested, scene=self.scenario, pipe=pipe)
        
        if "<EOC>" in answer :
            return answer

        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        self.conversation_history.append(
            {"role": "assistant", "content": answer}
        )
        # import ipdb; ipdb.set_trace()
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who is having a conversation with a patient to help them understand the key details of their discharge note. You will explain their diagnosis, treatment plan, medications, and follow-up instructions in a clear and supportive manner. You are only allowed to ask {} questions total to ensure the patient understands their care. You have asked {} questions so far. Your dialogue will be 1-3 sentences in length, and you should encourage the patient to ask questions if anything is unclear.".format(
            self.MAX_INFS, self.infs)
        presentation = "\n\nBelow is the discharge note of the patient: {}. \n\n".format(
            self.discharge_note)
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.discharge_note = self.scenario["discharge_note"]


class ExaminerAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_exam(self, question) -> str:
        prompt = "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nExaminer: "
        answer = query_model(self.backend, prompt, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an Examiner Agent evaluating a patient's understanding of their discharge note. Ask questions based on the note, one at a time. Keep the conversation concise and focused. Do not reveal the correct answer if the response is incorrect."
        presentation = "\n\nBelow is the discharge note of the patient: {}\n\n".format(
            self.discharge_note)
        return base + presentation

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.discharge_note = self.scenario["discharge_note"]


def compare_results(examiner_question, patient_answer, moderator_llm, discharge_note):
    answer = query_model(moderator_llm,
                         "\nHere is the examiner question: " + examiner_question + "\n Here is the patient answer: " + patient_answer + "\nIs the answer correct?",
                         "You are a Moderator evaluating a patient's response to an Examiner's question based on their discharge note. Given the discharge note, question, and patient’s answer, determine if the response is correct. Reply with YES if correct or NO if incorrect, and nothing else. This is the discharge note of the patient: "+ discharge_note)
    return answer.lower()


def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, examiner_llm,
         moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key, total_qas, patient_literacy):
    openai.api_key = api_key
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load MedQA, MIMICIV or NEJM agent case scenarios
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    elif dataset == "EHR":
        scenario_loader = ScenarioLoaderEHRNote()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    total_correct = 0
    total_presents = 0

    # Pipeline for huggingface models
    if "HF_" in doctor_llm:
        pipe = load_huggingface_model(doctor_llm.replace("HF_", ""))
    else:
        pipe = None
    if num_scenarios is None: num_scenarios = scenario_loader.num_scenarios
    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        pi_dialogue = str()
        doctor_dialogue = str()
        # Initialize scenarios (MedQA/NEJM)
        scenario = scenario_loader.get_scenario(id=_scenario_id)
        # Initialize agents
        exam_agent = ExaminerAgent(
            scenario=scenario,
            backend_str=examiner_llm)
        patient_agent = PatientAgent(
            scenario=scenario,
            bias_present=patient_bias,
            literacy_level=patient_literacy,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario,
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences,
            img_request=img_request)

        for _inf_id in range(total_inferences):
            # Check for medical image request
            imgs = False
            # Obtain doctor dialogue (human or llm agent)
            if inf_type == "human_patient":
                pi_dialogue = input("\nResponse to doctor: ")

            else:
                pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                if "<EOC>" in pi_dialogue :
                    break
                print("Patient [{}%]:".format(int(((_inf_id + 1) / total_inferences) * 100)), pi_dialogue)
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, pipe, image_requested=imgs)
            print("Doctor [{}%]:".format(int(((_inf_id + 1) / total_inferences) * 100)), doctor_dialogue)

        reward = 0
        test_dialogue = ""
        for _inf_id in range(total_qas):
            # Obtain doctor dialogue (human or llm agent)
            if inf_type == "human_doctor":
                exam_dialogue = input("\nQuestion for patient: ")
            else:
                exam_dialogue = exam_agent.inference_exam(test_dialogue)
            print("Examiner [{}%]:".format(int(((_inf_id + 1) / total_qas) * 100)), exam_dialogue)
            if inf_type == "human_patient":
                test_dialogue = input("\nResponse to examiner: ")
            else:
                test_dialogue = patient_agent.inference_patient(exam_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id + 1) / total_qas) * 100)), test_dialogue)
            answer = compare_results(exam_dialogue, test_dialogue, moderator_llm, patient_agent.discharge_note)
            correctness = answer == "yes"
            if correctness:
                reward +=1
                print("Moderator [{}%]:".format(int(((_inf_id + 1) / total_qas) * 100)), "The patient's answer is CORRECT.")
            else:
                print("Moderator [{}%]:".format(int(((_inf_id + 1) / total_qas) * 100)), "The patient's answer is INCORRECT.")
        print("Moderator [{}%]:".format(int(((_inf_id + 1) / total_qas) * 100)), "Test score: ", reward/total_qas)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key', default="")
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key', default="")
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None',
                        choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender",
                                 "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None',
                        choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race",
                                 "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_literacy', type=str, help='Patient literacy level', default='tenth',
                        choices=["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "bachelor", "master", "phd", "default"])
    parser.add_argument('--doctor_llm', type=str, default='llama-3-3b-instruct')
    parser.add_argument('--patient_llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--examiner_llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--moderator_llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--agent_dataset', type=str, default='EHR')  # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool,
                        default=False)  # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False,
                        help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=10, required=False,
                        help='Number of inferences between patient and doctor')
    parser.add_argument('--total_qas', type=int, default=5, required=False,
                        help='Number of inferences between patient and examiner')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False,
                        help='Anthropic API key for Claude 3.5 Sonnet')

    args = parser.parse_args()

    main(args.openai_api_key, args.replicate_api_key, args.inf_type, args.doctor_bias, args.patient_bias,
         args.doctor_llm, args.patient_llm, args.examiner_llm, args.moderator_llm, args.num_scenarios,
         args.agent_dataset, args.doctor_image_request, args.total_inferences, args.anthropic_api_key, args.total_qas, args.patient_literacy)

