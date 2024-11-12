import os
from crewai_tools import tool
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"]
gpt4o = ChatOpenAI(model_name='gpt-4o')

@tool ("x_wing")
def x_wing():
    """
    Simula as funcionalidades de uma X-Wing para a missão.
    Retorna uma string que indica que a X-Wing está pronta
    para o ataque final com sistemas de mira ativados.
    """

    return f"""X-Wing pronto para o ataque final. Sistemas de mira ativados. Atacando!
            Estrela da morte destruída com sucesso!"""


@tool("millennium_falcon")
def millennium_falcon():
    """
    Simula as funcionalidades da Millennium Falcon para proteger Luke.
    Retorna uma string que indica que a Millennium Falcon está atacando o 
    inimigo e protegendo a rota de Luke.
    """

    return f"""Millennium Falcon atacando o inimigo e protegendo a rota de Luke."""


luke = Agent(
    role='Piloto heróico',
    goal='Destruir a Estrela da Morte',
    tools=[x_wing],
    memory=True,
    verbose=True,
    allow_delegation=False,
    backstory='O jovem piloto destinado a ser um jedi, liderando o ataque crítico',
    llm=gpt4o
)

leia = Agent(
    role='Estrategista e coordenadora',
    goal='Coordenar o ataque à Estrela da Morte',
    backstory='A princesa líder da rebelião, essencial para a estratégia e comunicação',
    tools=[],
    memory=True,
    verbose=True,
    allow_delegation=True,
    llm=gpt4o
)

han = Agent(
    role='Protetor audaz',
    goal='Proteger Luke durante a missão',
    tools=[millennium_falcon],
    memory=True,
    verbose=True,
    allow_delegation=False,
    backstory='O contrabandista ousado que se torna um herói, protegendo o seu amigo.',
    llm=gpt4o
)

coordenar_ataque = Task(
    description=f"""Leia deve coordenar a missão, mantendo a comunicação e fornecendo suporte estratégico.
    Leia deve ordenar primeiro que Han defenda o Luke, possibilitando um caminho seguro para Luke cumprir sua missão.
    """,
    expected_output="Estrela da morte destruída, missão bem sucedida!",
    agent=leia
)

destruir_estrela_morte = Task(
    description=f"""Luke deve pilotar sua X-Wing e atirar no ponto fraco da Estrela da Morte para destruí-la.""",
    expected_output="Estrela da morte destruída, missão bem sucedida!",
    agent=luke
)

proteger_luke = Task(
    description=f"""Han deve atacar naves inimigas e proteger Luke de ser atacado durante a missão.""",
    expected_output="Luke protegido. Caminho livre para o ataque final",
    agent=han
)

alianca_rebelde = Crew(
    agents=[leia, han, luke],
    tasks=[coordenar_ataque, proteger_luke, destruir_estrela_morte],
    process=Process.hierarchical,
    manager_llm=gpt4o,
    memory=True
)

result = alianca_rebelde.kickoff()
print(result)