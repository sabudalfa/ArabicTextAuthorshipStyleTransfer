from llm import *

jais_loader = LLMLoader(
    '/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(),
)
text_generator = TextGenerator.from_llm_loader(jais_loader)
text_generator.generate(
    ["This is a prompt"],
    max_new_tokens=4000,
)
