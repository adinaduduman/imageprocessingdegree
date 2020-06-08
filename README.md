# imageprocessingdegree
Input: Imaginea cu plangerea scrisa de mana
Output: Un procent corespunzator fiecarei categorii prestabilite

Algoritmul este format din mai multe parti:

1. Preprocesarea: consta in pregatirea imaginii pentru a recunoaste cuvintele si cuprinde urmatoarele etape:
	- citirea imaginii
	- ajustarea contrastului
	- segmentarea liniilor
	- segmentarea cuvintelor

2. Algoritmul de recunoastere a cuvintelor
	-folosind un model deja antrenat al unei retele neuronale, se va recunoaste cuvantul pe baza cuvantului segmentat
	

3. Postprocesarea: avand lista de cuvinte recunoscute se vor urma pasii
	- folosind libraria pyspellchecker se va aduce cuvantul la forma ortografica cea mai asemanatoare
	- se foloseste un api care primeste ca input un cuvant si ca output ofera cuvinte din campul semantic al cuvantului
	- pentru fiecare cuvant din lista, se face un call la acel api din care se iau primele 2 cuvinte (cuvintele cu sens cel mai asemanator)
	- folosind algoritmul Cosine similarity si modelul Glove (Global Vectors for Word Representation) se verifica cuvintele din categoria prestabilita
	si se calculeaza procentul de similaritate intre textul dat ca input si categoriile prestabilite (descrise mai jos)

4. Categoriile prestabilite contin initial cate 4 cuvinte din campul semantic al fiecarui element.
De exemplu, pentru cateogria "fruits" contine 4 cuvinte din campul semantic al fructelor. ("lemon orange strawberry apple")
Iar categoria "stars" va cuprinde cuvintele "stars comet alien sky".
Cuvintele din categorii se vor compara cu cele date la input (folosind ultimul pas al postprocesarii)
