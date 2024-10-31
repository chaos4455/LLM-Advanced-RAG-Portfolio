# 🌟 Portfolio  -  Implementação de RAG com python e sqlite on scratch🌟

![RAG Badge](https://img.shields.io/badge/RAG_Implementation-Complete-brightred)
![Version](https://img.shields.io/badge/Version-1.0-brightred)
![Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![License](https://img.shields.io/badge/License-MIT-brightred)

# **Elias Andrade** 🚀

## 🌟 Sobre o Projeto de RAG

Atualmente, estou desenvolvendo um projeto inovador de **Retrieval-Augmented Generation (RAG)**, que visa aprimorar a geração de respostas em sistemas de inteligência artificial utilizando a recuperação de informações de maneira eficaz. Este projeto combina a capacidade de **Geração de Texto** com técnicas de **Busca Inteligente**, permitindo que a IA não apenas produza conteúdo, mas também o faça de maneira fundamentada e informada.

### 🛠️ Tecnologias Utilizadas

- **Modelos de Linguagem**: Integração de LLMs (Modelos de Linguagem de Grande Escala) para melhorar a qualidade da geração de texto.
- **Sistemas de Recuperação**: Implementação de algoritmos de recuperação de informações que otimizam a busca e a filtragem de dados relevantes.
- **Banco de Dados**: Utilização de bancos de dados NoSQL e SQL para armazenar e gerenciar os dados utilizados nas operações de recuperação e geração.
- **Frameworks de Machine Learning**: Aplicação de bibliotecas como TensorFlow e PyTorch para treinar e otimizar modelos.

# 📚 Recuperação de Conhecimento e Arquiteturas de RAG (Retrieval-Augmented Generation) em IA

A capacidade de integrar vastas quantidades de dados e gerar respostas coerentes e úteis é um dos maiores avanços recentes em IA, e para isso, métodos como **RAG** (Retrieval-Augmented Generation), bancos vetoriais e embeddings se tornaram cruciais. Neste documento, exploramos esses conceitos de maneira profunda, mostrando como eles operam em conjunto e são fundamentais para o processamento e geração de informações.

---

## 📌 O que é RAG?

**RAG (Retrieval-Augmented Generation)** é uma técnica que combina modelos de recuperação de informações (retrieval) e modelos gerativos (generation) para obter respostas contextualmente ricas. Em vez de depender exclusivamente do modelo generativo para produzir informações, RAG busca dados específicos em um banco de conhecimento pré-indexado e usa esse conteúdo como contexto para a geração de respostas. Esse processo melhora a acurácia e a relevância da resposta, já que o modelo gera respostas com base em conhecimento extraído de dados factuais, e não apenas em seu treinamento.

Em resumo:
- 🔍 **Recuperação (Retrieval):** Busca informações relevantes em um banco de dados ou banco vetorial.
- 📝 **Geração (Generation):** Usa essas informações para produzir respostas detalhadas e contextuais.
- ✅ **Benefício:** Maior precisão e consistência, especialmente em tarefas que exigem conhecimento atualizado ou específico.

<img width="841" alt="Screen-Shot-2018-04-25-at-13 21 44" src="https://github.com/user-attachments/assets/4c3a9362-101c-41e6-b5e0-45570ae190df">

---

## 📊 O que é um Banco Vetorial?

Um **banco vetorial** é uma estrutura de dados que armazena informações em formato de vetores numéricos em vez de texto puro. Isso possibilita que textos, imagens ou qualquer dado seja transformado em vetores que capturam seu significado ou contexto de forma matemática, utilizando **embeddings** para representar semântica e relevância. Esses vetores permitem operações como **busca por similaridade**, onde conteúdos semelhantes são identificados com rapidez e eficiência através da proximidade vetorial.

![How-Embeddings-Work](https://github.com/user-attachments/assets/e7cac95a-536b-4db3-ba26-b25246f9591c)

### 🛠️ Como Funciona um Banco Vetorial?
1. **Conversão para Vetores:** Dados textuais são convertidos para vetores utilizando modelos de embeddings, como o BERT, ou técnicas mais modernas e robustas, como o **transformer embeddings**.
2. **Armazenamento:** Esses vetores são indexados e armazenados em um banco vetorial (por exemplo, Pinecone, Weaviate, ou bases de dados como Qdrant).
3. **Busca por Similaridade:** Quando uma consulta é feita, ela também é convertida em vetor e comparada com os vetores no banco, retornando os mais próximos, ou seja, os mais contextualmente relevantes.

Os bancos vetoriais, portanto, permitem realizar buscas semânticas profundas, encontrando dados correlatos de forma muito mais sofisticada e precisa do que buscas por palavras-chave.

![maxresdefault](https://github.com/user-attachments/assets/3b3cf80f-629d-4feb-b7ef-34e7228bba27)

---

## 🧬 Entendendo Vetores e Embeddings no Contexto de IA

### 🔹 O que é um Vetor?

Um **vetor**, no contexto da IA, é uma representação numérica que captura as características de um dado (texto, imagem, etc.) de forma que ele possa ser manipulado matematicamente. Os vetores são a base para algoritmos de busca, recomendação e muitos outros processos em IA, pois contêm as informações essenciais sobre similaridade e contexto. 

Por exemplo, ao transformar uma frase em vetor, o modelo de IA capta semântica, tom e intenção, de modo que frases semelhantes (em significado) tenham representações vetoriais próximas.

![sualizations-of-the-SVD-based-node2vec-embeddings-first-row-and-original-node2vec_Q320](https://github.com/user-attachments/assets/48335866-c039-4165-a292-c81cb2559b17)

### 🔹 O que é um Embedding?

Um **embedding** é a técnica utilizada para transformar dados complexos, como frases e imagens, em vetores de alta dimensionalidade. O embedding é gerado por modelos de aprendizado profundo e serve como uma **"impressão digital"** do conteúdo, permitindo comparações precisas. Esses embeddings são criados por modelos treinados para entender nuances e relações entre palavras e frases, o que significa que **embeddings capturam o significado contextual** e permitem operações avançadas de busca semântica.

- **Exemplo:** Em um modelo de embeddings, palavras como "carro" e "automóvel" gerariam vetores muito próximos, enquanto palavras como "carro" e "banana" estariam distantes no espaço vetorial.

  ![0_hkbsIc6g6u9DWZMH](https://github.com/user-attachments/assets/9686d66f-2120-4b28-84a7-8c67ae61a7bb)


### 🔹 Criação e Utilização de Embeddings

1. **Modelos de Embeddings:** Modelos de linguagens, como BERT, RoBERTa e GPT, são usados para gerar embeddings. Esses modelos são pré-treinados em grandes volumes de texto para captar significado e contexto das palavras.
2. **Aplicação dos Embeddings:** Uma vez que um dado (ex. frase) é convertido para um embedding, ele pode ser armazenado em um banco vetorial e comparado com outros embeddings, permitindo buscas e análises contextuais sofisticadas.
3. 
![fig3](https://github.com/user-attachments/assets/7f40cfe6-fa23-4583-912d-51ad7d892dfc)

---

## 🚀 Aplicações e Benefícios do Uso de RAG e Bancos Vetoriais

O uso de RAG e bancos vetoriais têm aplicações práticas extensas e profundas, especialmente em setores que requerem processamento de linguagem natural, busca avançada e geração de texto. Algumas aplicações incluem:

- **Chatbots Inteligentes e Copilotos de Código:** Melhoram as respostas ao integrar dados externos em tempo real.
- **Sistemas de Recomendação:** Identificam conteúdo e produtos de interesse com base em similaridade semântica.
- **Suporte ao Cliente Automatizado:** Oferecem respostas precisas com base em uma base de conhecimento.
- **Recuperação de Documentos e Pesquisa Jurídica:** Localizam documentos complexos e específicos rapidamente, aumentando a eficiência.
- **Geração de Conteúdo:** Utilizam dados específicos e embasados para gerar textos, estudos e análises de forma automatizada e precisa.

![image10-ebe747ac9f2e03dba758f1ed3ea7e82c](https://github.com/user-attachments/assets/d00a7120-720c-400d-9059-d7b06ee014b8)

### 🔥 Vantagens
- **Alta Relevância:** Dados são recuperados com base em similaridade semântica, não apenas por palavras-chave.
- **Escalabilidade:** Bancos vetoriais são altamente escaláveis e eficientes para consultas em tempo real.
- **Precisão Contextual:** Embeddings melhoram a compreensão do modelo sobre significado e contexto, aumentando a qualidade da resposta gerada.

---
![3d-vector-representation eabfb5ea](https://github.com/user-attachments/assets/3b9888cb-b067-4f95-9330-464daae7a7b5)


## 🧩 Conclusão

A utilização de **RAG**, **bancos vetoriais** e **embeddings** transforma a maneira como IA processa, armazena e gera informações. Esses conceitos formam o núcleo de sistemas modernos de IA, permitindo buscas e geração de dados altamente contextualizadas, relevantes e rápidas. À medida que os modelos e técnicas de geração de embeddings e busca vetorial avançam, as aplicações se tornam mais sofisticadas e impactantes, fornecendo uma base sólida para sistemas de inteligência artificial que podem interagir com dados complexos de maneira eficiente e significativa.

![vectors](https://github.com/user-attachments/assets/28c34267-e825-4567-8637-b5182aabb96e)

### 📈 Objetivos do Projeto

- **Aprimorar a Precisão**: Garantir que a geração de respostas seja não apenas coerente, mas também precisa, utilizando informações extraídas de fontes confiáveis.
- **Otimização de Processos**: Automatizar o fluxo de trabalho, desde a recuperação até a geração, para garantir eficiência e rapidez nas respostas.
- **Experiência do Usuário**: Melhorar a interação do usuário com sistemas de IA, oferecendo respostas mais relevantes e contextualizadas.

### 💡 Aplicações Práticas

- **Chatbots Avançados**: Desenvolvimento de chatbots que oferecem respostas mais ricas e baseadas em dados reais.
- **Suporte ao Cliente**: Implementação de sistemas de suporte que utilizam RAG para fornecer soluções rápidas e informadas.
- **Geração de Conteúdo**: Criação de ferramentas que ajudam na redação de textos, relatórios e documentação com base em informações recuperadas.

### 🚀 Vamos Conversar!

Estou aberto a parcerias e discussões sobre o projeto de RAG e suas aplicações. Se você deseja saber mais ou explorar colaborações, entre em contato!

📅 **Agende uma reunião:** [Clique aqui para agendar uma call de 30 minutos comigo no Calendly!](https://calendly.com/oeliasandrade/30min)

---

## 🌐 Informações de Contato

- **E-mail:** elias.andrade@email.com
- **LinkedIn:** [linkedin.com/in/eliasandrade](https://www.linkedin.com/in/itilmgf)


## 🚀 Introdução ao RAG (Retrieval-Augmented Generation)

Como especialista em tecnologia e IA, estou animado para compartilhar minha experiência na implementação de um sistema de **Retrieval-Augmented Generation (RAG)** do zero. O RAG combina as capacidades de geração de linguagem de modelos de linguagem (LLMs) com a busca e recuperação de informações, permitindo que sistemas autônomos tomem decisões informadas com base em dados relevantes.

### O que é RAG? 🤖

O RAG é uma abordagem inovadora que integra a recuperação de documentos e a geração de texto. Ao utilizar um modelo de linguagem avançado, como o **Google Gemini**, em conjunto com um banco de dados **SQLite**, criei uma arquitetura que pode acessar informações em tempo real e gerar respostas mais precisas e contextualizadas.

**Benefícios do RAG:**
- **Melhoria da Precisão**: O RAG utiliza informações específicas para gerar respostas mais acuradas, superando as limitações de modelos que operam apenas com aprendizado prévio.
- **Atualização em Tempo Real**: Sistemas que utilizam RAG podem se adaptar e aprender com novos dados, garantindo relevância contínua.
- **Interação Mais Rica**: Proporciona interações mais informativas e úteis, especialmente em aplicações corporativas e de suporte ao cliente.

---

## 📊 Arquitetura da Implementação

### 1. **Arquitetura Geral**

Para construir o meu sistema RAG, adotei a seguinte arquitetura:

- **Frontend**: Interface de usuário simples para interagir com o sistema.
- **Backend**: 
  - **FastAPI**: Para gerenciar requisições de API.
  - **Google Gemini**: Para geração de linguagem.
  - **SQLite**: Para armazenamento de dados e recuperação de informações.

### 2. **Fluxo de Dados**

1. **Entrada do Usuário**: O usuário envia uma consulta através da interface.
2. **Recuperação de Dados**: O sistema utiliza o SQLite para buscar informações relevantes com base na consulta.
3. **Geração de Resposta**: O Google Gemini processa os dados recuperados e gera uma resposta que é retornada ao usuário.

### 3. **Implementação do RAG**

#### Masterização do RAG

- **Data Collection**: Coletar dados relevantes de diferentes fontes para alimentar o banco de dados SQLite.
- **Database Schema**: Estruturar o SQLite com tabelas apropriadas para armazenar documentos e suas representações vetoriais.
- **Document Retrieval**: Implementar algoritmos de recuperação que possam consultar e filtrar documentos de forma eficiente.
- **Text Generation**: Integrar o Google Gemini para geração de texto, utilizando os dados recuperados como contexto.

---

## 🧩 Componentes Principais da Implementação

### 1. **FastAPI** 🚀

Utilizei o FastAPI para criar uma API RESTful, que facilita a interação entre o frontend e o backend. Com FastAPI, consigo atender a requisições rapidamente, oferecendo respostas em tempo real. O suporte a operações assíncronas garante que o sistema permaneça responsivo, mesmo sob carga.

### 2. **SQLite** 🗄️

O SQLite foi escolhido por sua leveza e facilidade de integração. Ele armazena os dados que são posteriormente consultados pelo sistema RAG. A estrutura do banco de dados foi otimizada para permitir buscas rápidas e eficientes. A combinação do SQLite com um sistema de recuperação de documentos oferece um ambiente ideal para o RAG.

### 3. **Google Gemini** 🤖

Integrei o Google Gemini para a geração de texto. A capacidade de processar informações e gerar respostas coerentes e contextualizadas é fundamental para o sucesso do RAG. O Google Gemini se destaca pela sua eficiência em compreender o contexto e produzir respostas mais alinhadas às expectativas do usuário.

---

## 🔍 Aplicações do RAG em Sistemas Super Inteligentes

A implementação de RAG possui diversas aplicações em sistemas autônomos e inteligentes:

1. **Suporte ao Cliente**: Sistemas de atendimento automatizado que utilizam RAG podem fornecer respostas mais precisas e personalizadas com base em consultas frequentes.
2. **Análise de Dados**: Sistemas que analisam grandes volumes de dados podem gerar relatórios com insights valiosos, aproveitando a recuperação de informações.
3. **Assistência Pessoal**: Assistentes virtuais que utilizam RAG podem aprender continuamente e melhorar a qualidade das interações com os usuários.

---

## 🏆 Resultados e Benefícios da Implementação

- **Eficiência Aumentada**: A implementação do RAG melhorou significativamente a eficiência do sistema na geração de respostas precisas.
- **Satisfação do Usuário**: Os usuários relatam uma experiência mais rica e informativa, resultando em um aumento na satisfação.
- **Escalabilidade**: A arquitetura permite que o sistema escale facilmente à medida que novos dados são adicionados, mantendo a relevância e a precisão.

---

## 🔒 Conclusão

Implementar um sistema RAG do zero usando Python, SQLite e Google Gemini foi um desafio gratificante que resultou em um produto final robusto e eficaz. Este projeto não apenas aprimorou minhas habilidades técnicas, mas também me proporcionou uma compreensão profunda das aplicações práticas do RAG em sistemas super inteligentes e autônomos.

### Veja abaixo vários prints de testes e validações do framework e sua aplicação na prática! 📸👇

---

### 📊 **Manipulação de Vetores e Conversão**
- **Exemplo de Vetores**: Mostramos como criar, manipular e converter vetores em formatos utilizáveis para a IA.
- **Transformações em DataFrame**: Demonstrações sobre como transformar vetores em DataFrames para uma análise mais robusta.

---

### 🔧 **Engenharia de Prompt e Embedding do RAG**
- **Estratégias de Engenharia de Prompt**: Exemplos práticos de como construir prompts eficazes que maximizam a geração de resultados de qualidade.
- **Embeddings**: Visualizações dos embeddings gerados, destacando como eles se integram ao processo de RAG.

---

### ⚙️ **Pipeline de Manipulação**
- **Automação de Validação**: Exibições de automações que garantem a integridade do banco de vetores e da estrutura.
- **Manuseio de Vetores**: Prints de como os vetores são manipulados ao longo do pipeline, desde a entrada até a geração de saídas.

---

### 🔍 **Validação e Testes**
- **Testes de Estrutura**: Demonstrações da validação da estrutura dos dados em diferentes cenários.
- **Resultados de Performance**: Análises de performance e precisão dos resultados gerados, com prints de testes em ação.

![RAG Success](https://img.shields.io/badge/Success-100%25-success)
![Ready for Production](https://img.shields.io/badge/Ready_for_Production-Yes-brightgreen)
![Continuous Improvement](https://img.shields.io/badge/Continuous_Improvement-Ongoing-brightred)


<img width="302" alt="Cursor_4LDeqVo0pr" src="https://github.com/user-attachments/assets/078acf13-1c4e-4f21-9df5-294522b90725">
<img width="662" alt="chrome_78JwlYhifp" src="https://github.com/user-attachments/assets/80d04b3f-5243-4d71-b90e-abd3f7c32332">
<img width="653" alt="chrome_IzoIqHpStU" src="https://github.com/user-attachments/assets/5419db6f-3cc0-4946-b6e7-bf9011f32909">
<img width="650" alt="chrome_kjly2aWazj" src="https://github.com/user-attachments/assets/444eaea7-a091-4d92-bbc3-9f37d5d6c3c9">
<img width="701" alt="Cursor_1yrTzUjocF" src="https://github.com/user-attachments/assets/fdfcc916-ac96-4fc1-958f-e886ce8a80cd">
<img width="327" alt="Cursor_xNVxjl1MZd" src="https://github.com/user-attachments/assets/259f7e32-4fb8-46ee-a1db-b05081147c40">
<img width="513" alt="Cursor_xWCL6YJpaS" src="https://github.com/user-attachments/assets/4a937382-ea5e-48d3-a918-b31ff9e1705f">
<img width="550" alt="Cursor_Ta7wRpvDW4" src="https://github.com/user-attachments/assets/352c3637-dd72-4147-a9ea-a73c06dcc0bc">
<img width="448" alt="Cursor_ycLq73wiA2" src="https://github.com/user-attachments/assets/e5b080fa-285e-4e95-bb38-19344dd74ce4">
<img width="252" alt="Cursor_AQldZoNP42" src="https://github.com/user-attachments/assets/3bce2551-fa4c-40ca-b37d-4e9072441987">
<img width="304" alt="Cursor_jALVVO6DCN" src="https://github.com/user-attachments/assets/1e1f7650-085e-45ab-a7a8-df24daba1d30">
<img width="209" alt="Cursor_7lmqNMF6YQ" src="https://github.com/user-attachments/assets/ca80c102-ef39-43f3-aeec-e4135af25c47">
<img width="178" alt="Cursor_qvUBEymNLA" src="https://github.com/user-attachments/assets/083a1577-3b28-45fd-aa26-1ffc97e4aa5c">
<img width="178" alt="Cursor_6aaqcxffdC" src="https://github.com/user-attachments/assets/092cb7e6-9ca5-43d0-ae31-af2b8545ce34">
<img width="188" alt="Cursor_QtddD6m3hD" src="https://github.com/user-attachments/assets/f2ecb509-25a8-42ee-86a2-38e3fa524afa">
<img width="202" alt="Cursor_jmzYxitvdn" src="https://github.com/user-attachments/assets/610b4b53-78a2-49c9-ac20-447cd9ed4168">
<img width="169" alt="Cursor_f6dDIRXWFv" src="https://github.com/user-attachments/assets/a78e4455-bd4d-4a48-8cbd-eb4d5e85dac9">
<img width="330" alt="Cursor_Ye5rRrUhgK" src="https://github.com/user-attachments/assets/c33c4df0-a8bb-4dac-b9f8-8470e7e3bd6a">
<img width="958" alt="Cursor_F1FhyT94lW" src="https://github.com/user-attachments/assets/5271a85e-b804-4693-af61-47b6926e34c7">
<img width="191" alt="Cursor_eeWMOuyrHI" src="https://github.com/user-attachments/assets/f1388508-038b-407a-b710-6f9291ab3a91">
<img width="363" alt="Cursor_DJlvv30gza" src="https://github.com/user-attachments/assets/bf2eed48-9bf6-4599-9949-72ef01e55cfb">
<img width="303" alt="Cursor_ceShCh9msp" src="https://github.com/user-attachments/assets/6bdcabed-8888-4700-abcf-cf15ad4c0a5a">
<img width="192" alt="Cursor_4Tecd3fTo0" src="https://github.com/user-attachments/assets/09e8ffd0-3e0b-45cd-a292-57c6903b5133">
<img width="286" alt="Cursor_FCv5HrN4Lz" src="https://github.com/user-attachments/assets/59f4b7b9-5390-476b-8ad7-364ec26d0f4c">
<img width="327" alt="Cursor_IlRYDbpzIA" src="https://github.com/user-attachments/assets/f9c479cd-1f3f-4e38-b32a-ba70097323ed">
<img width="292" alt="Cursor_Se3sp5ydlL" src="https://github.com/user-attachments/assets/656a46e6-995e-4438-94de-b75c93e5988d">
<img width="642" alt="Cursor_Ed9uqT2TbE" src="https://github.com/user-attachments/assets/0468ae7e-a60b-4b52-a509-daf03abb5d1b">
<img width="689" alt="Cursor_PcRx3xCli3" src="https://github.com/user-attachments/assets/d60076e0-8321-43bc-822e-322e2d519d07">
<img width="654" alt="Cursor_aABgOKDX7p" src="https://github.com/user-attachments/assets/347d1956-4071-4bee-a803-c394ac31e9a8">
<img width="468" alt="Cursor_ATAvkcR48r" src="https://github.com/user-attachments/assets/4457e52f-0adc-4bb5-afce-ed3897ce699d">
<img width="675" alt="Cursor_St0DdF5qbk" src="https://github.com/user-attachments/assets/4ac6370e-949e-4bc6-a069-3c33ac7a87aa">
<img width="409" alt="Cursor_nCWNnRzP95" src="https://github.com/user-attachments/assets/64f0491c-78b0-4194-8dbc-502d75b12eb8">
<img width="221" alt="Cursor_byyIKdx0Ns" src="https://github.com/user-attachments/assets/b65578db-0e12-4470-b212-6c8a45ba8bc2">
<img width="201" alt="Cursor_wP3oSGAVXX" src="https://github.com/user-attachments/assets/0f7a2c34-8bbb-403c-9864-992de2f87d8a">
<img width="372" alt="Cursor_Bx5bWNrqmA" src="https://github.com/user-attachments/assets/d0a7aa8c-6a36-448b-8c93-b4385b2a9ede">
<img width="682" alt="Cursor_wqa5mIiI8h" src="https://github.com/user-attachments/assets/5493e6b5-6f57-4ede-938c-377b8ebd778f">
<img width="325" alt="Cursor_mZrQaZqcZ8" src="https://github.com/user-attachments/assets/5a320bd7-bebf-4aa4-8915-bc1ed32ea23b">
<img width="302" alt="Cursor_JoL4msLzsL" src="https://github.com/user-attachments/assets/bdeb2dfc-dcf0-49a0-9d11-c6d65b59e14e">
<img width="416" alt="Cursor_XZkAP9KmIz" src="https://github.com/user-attachments/assets/363a8945-0870-4db1-b831-132b88531516">
<img width="229" alt="Cursor_3eB1ytL4PE" src="https://github.com/user-attachments/assets/6c3d180e-1922-4c42-a2f4-75d6f52f08ea">
<img width="421" alt="Cursor_hERaj6De1D" src="https://github.com/user-attachments/assets/e947f510-93d7-47d1-a303-321f64242248">
<img width="415" alt="Cursor_18XvYQ3ORz" src="https://github.com/user-attachments/assets/86e6a81b-cd33-4961-9047-c9fb3e45b82b">
<img width="166" alt="Cursor_U03E5ApUpR" src="https://github.com/user-attachments/assets/3d6ef3e5-1ba1-4e3c-b42c-173901eba1d9">
<img width="204" alt="Cursor_kfKM2BIiKg" src="https://github.com/user-attachments/assets/0cb81f1b-ac9d-4899-bdd4-33e9fb51d488">
<img width="520" alt="Cursor_eUoUPo6B6c" src="https://github.com/user-attachments/assets/5d7d762f-1c00-4a41-aeff-5156993cf366">
<img width="421" alt="Cursor_Ga8sKhgvK8" src="https://github.com/user-attachments/assets/375b96eb-72ce-4204-8092-288053a0538e">
<img width="1065" alt="Cursor_PmEXl4CFEk" src="https://github.com/user-attachments/assets/9f88d70b-65d7-48ee-aa4d-975b8a07afc9">
<img width="303" alt="Cursor_OjzEq1pVjK" src="https://github.com/user-attachments/assets/5bff7ae1-94b8-48b1-80e2-a1c95a81f261">
<img width="240" alt="Cursor_wTzxjlVU5V" src="https://github.com/user-attachments/assets/d666e137-50e6-457d-a375-5ba662cc01cc">
<img width="295" alt="AsPowerBar_31RjNtOPPr" src="https://github.com/user-attachments/assets/4481cb70-c034-421a-94d2-c92cc569e28d">
<img width="284" alt="Cursor_x2FPxXmTzM" src="https://github.com/user-attachments/assets/0c8bce5a-a6de-4d15-99fa-027416dd378e">
<img width="492" alt="Cursor_hVDIUQgA7m" src="https://github.com/user-attachments/assets/5312cc28-095e-43e5-a46a-fe7fb7acbf95">
<img width="299" alt="Cursor_J9NiGmD3ob" src="https://github.com/user-attachments/assets/5b8496e3-a51c-4e66-8d2d-42dd9dd22c0b">
<img width="690" alt="Cursor_S0oOmYkmFn" src="https://github.com/user-attachments/assets/c593dcd3-ab9e-4f81-9250-c6c2422fbf1c">
<img width="665" alt="Cursor_x5w1qqKPgx" src="https://github.com/user-attachments/assets/9a2571e2-a1be-4bd3-a581-65ea18bc81e8">
<img width="291" alt="Cursor_nFpxTFUxIG" src="https://github.com/user-attachments/assets/45f0c217-7c73-4f66-83ac-72ff6bfe7e82">
<img width="282" alt="Cursor_BiCCVvG3Gl" src="https://github.com/user-attachments/assets/29b4905d-e48b-4664-8d38-59f216e78bdb">
<img width="283" alt="Cursor_e71H61HiXi" src="https://github.com/user-attachments/assets/4cd8d1ad-5d4b-40bb-b32d-dcf81d2d04c9">
<img width="443" alt="Cursor_jUj6vJbPY2" src="https://github.com/user-attachments/assets/575e1359-b778-4f78-8ed7-4c5c7a26c857">
<img width="312" alt="Cursor_eDNG5h6HyY" src="https://github.com/user-attachments/assets/5c80464a-eec0-420f-b99a-fa8ca42ff31b">
<img width="259" alt="Cursor_uKs1hdKPqH" src="https://github.com/user-attachments/assets/6eddb268-7fdc-4f4a-ae83-a7bd527f2ef4">
<img width="779" alt="Cursor_LEEXbd354Q" src="https://github.com/user-attachments/assets/3fcae5f7-9790-4e6c-8234-6f8873236e39">
<img width="465" alt="Cursor_ozRuXazjwB" src="https://github.com/user-attachments/assets/4303e874-2584-427d-bba4-865aa5d0f9d3">
<img width="468" alt="Cursor_irwl9MOkcb" src="https://github.com/user-attachments/assets/5ae708a0-46ac-444a-9251-beb69b32ef9b">
<img width="579" alt="Cursor_WOlvjsoGiC" src="https://github.com/user-attachments/assets/16bbe7ca-628f-46a2-9a01-a12ffc10a3d2">
<img width="448" alt="Cursor_lkRbsDCDiM" src="https://github.com/user-attachments/assets/14df761f-977b-4c6e-8583-80b0678d0ea5">
<img width="456" alt="Cursor_C9YH0tkqcA" src="https://github.com/user-attachments/assets/fb232858-fff6-42a0-aae9-5b1d06c89e76">
