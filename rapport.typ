// Configuration

#let project_name = "Techniques d'optimisation parallèles: TP Noté"

#let conf(title: none, project_name: none, authors: (), abstract: [], doc) = {
  	// Set and show rules from before.
	set align(center)
	text(17pt)[
    		*#title #project_name*
	]

	set text(
		lang: "fr",
	)

	let count = authors.len()
	let ncols = calc.min(count, 3)
	grid(
		columns: (1fr,) * ncols,
		row-gutter: 24pt,
		..authors.map(author => [
			#author.name \
			#author.affiliation \
		]),
	)

	set align(left)
	par(justify: false)[
		*Résumé* \
		#abstract
	]

	set align(left)
	doc
}

#show link: this => {
	let show-type = "box" // "box" or "filled", see below
	let label-color = green
	let default-color = blue
	
	if show-type == "box" {
		if type(this.dest) == label {
			// Make the box bound the entire text:
			// set text(bottom-edge: "bounds", top-edge: "bounds", stroke: label-color)
			// text(this, stroke: label-color)
			text(label-color, [#this])
			// box(this, stroke: label-color + 1pt)
		} 
		else {
			//set text(bottom-edge: "bounds", top-edge: "bounds", stroke: default-color)
			text(default-color, [#this])
			// box(this, stroke: default-color + 1pt)
		}
	} 
	else if show-type == "filled" {
		if type(this.dest) == label {
			text(this, fill: label-color)
		} 
		else {
			text(this, fill: default-color)
		}
	} else {
		this
	}
}


// Set page style
#set list(indent: 1cm)
#set page(
	margin: 1.5cm,
	header: [
		#set align(right + horizon)
		#text(10pt)[
			#emph(project_name)
		]
	], 
	footer: [
		#set align(right)
		#context counter(page).display(
			"1 / 1",
			both: true,
		)
	],
)

// Show a snippet of code inside a coding-styled block
#let code_snippet(code: none, title: none, tabwidth: 4, lang: "c") = {
	let spacing = " "
	let x = 1
	let replaced_code = ""
	while x < tabwidth {
		spacing = spacing + " "
		x = x + 1
	}
	for char in code {
		if char == "\t" {
			replaced_code = replaced_code + spacing
		} else {
			replaced_code = replaced_code + char
		}
	}
	set align(left)
	text(11pt)[
		#title
	]
	set align(left)
	text(9pt)[
		#raw(replaced_code, lang: lang)
	]
}

#show: conf.with(
	title: [
		Techniques d'optimisation parallèles: Étude et amélioration des performances de la multiplication de matrices
	],
	project_name: [],
	authors: (
		(
			name: "BAUMANN Pierre",
			affiliation: "Master 1 CHPS",
		),
	),
	abstract: [
		Le but du TP est d'étudier et d'améliorer les performances de l'opération 
		$C #sym.arrow.l #sym.alpha * A * B + #sym.beta * C$, avec $A$, $B$ et $C$ des matrices, 
		et $#sym.alpha$ et $#sym.beta$ des scalaires.
		Nous allons analyser les performances de l'implémentation naïve, puis nous présenterons et testerons diverses méthodes d'optimisation.
	]
)

#set heading(numbering: "1.")
#outline(indent: 1cm)

= Introduction

La multiplication de matrices est une opération fondamentale en informatique, utilisée dans de nombreux domaines tels que l'apprentissage automatique, la simulation numérique et le traitement d'images.
Dans ce TP, nous allons étudier et améliorer les performances de l'opération $C #sym.arrow.l #sym.alpha * A * B + #sym.beta * C$, avec $A$, $B$ et $C$ des matrices, et $#sym.alpha$ et $#sym.beta$ des scalaires.#footnote("L'implémentation de base possède un bug, mais j'ai gardé ce bug lors des optimisations")

Nous allons d'abord faire une étude, en strong scaling, de l'impact des layouts mémoire sur les performances.

Ensuite, nous allons implémenter un algorithme de cache blocking afin de maximiser la localité spatiale et temporelle, et étudier les gains obtenus.

De plus, nous allons porter l'algorithme sur GPU pour le comparer à l'implémentation CPU.

Enfin, nous allons conclure sur les résultats obtenus.


= Layouts

Les layouts mémoire sont des termes utilisés pour décrire la manière dont les données sont organisées en mémoire pour des tableaux multidimensionnels. Ici, en 2 dimensions car il s'agit de matrices.

Donc pour ranger les éléments d'une matrice en mémoire, nous avons 2 choix, en row-major (Les lignes sont contiguës en mémoire) ou en column-major (Les colonnes sont contiguës en mémoire).

#figure(
	grid(
		columns: 1,
		rows: 3,
		row-gutter: 0.2cm,

		$ A = mat(
			1, 2, 3;
			4, 5, 6;
			7, 8, 9;
		) $
		,
		$ A_"row-major" = mat(
			1, 2, 3, 4, 5, 6, 7, 8, 9
		) $
		,
		$ A_"column-major" = mat(
			1, 4, 7, 2, 5, 8, 3, 6, 9
		) $
	)
	, caption: "Exemple de matrice et ses représentations en row-major et column-major"
)

Kokkos nous permet de choisir le layout mémoire de nos matrices avec les vues #emph[Kokkos::LayoutRight] pour le row-major et #emph[Kokkos::LayoutLeft] pour le column-major.

Comme nous avons 3 matrices, il y a $2^3 = 8$ combinaisons de layouts possibles (soit LayoutRight, soit LayoutLeft pour $A$, $B$, $C$)

D'abord, je voulais faire une étude en strong scaling sur une matrice de taille $1000 * 1000$, mais j'ai vite remarqué que cela prennait anormalement trop de temps.
J'ai donc d'abord testé sur des plus petites matrices, de taille $200 * 200$

#figure(
	image(
		"results/strong_scaling_layout_all.svg",
		width: 100%
	),
	caption: "Étude en strong scaling de l'impact des layouts mémoire sur les performances sur des matrices 200x200. Le plus bas est mieux."
)

Nous pouvons immédiatement remarquer que 2 combinaisons produisent des résultats abhérents, celles qui ont $A$ en #emph[LayoutLeft] et $B$ en #emph[LayoutRight].
En effet, ce sont les seules combinaisons où le temps d'exécution augmente avec le nombre de threads.
Cela semble indiquer un problème de faux partage ou de contention des threads.

Si nous reprenons le code de notre kernel:
#code_snippet(
code: "
/* Parallel */ for (int i = 0; i < n; i++) {
	/* Sequential */ for (int j = 0; j < n; j++) {
		double acc = 0.0;
		/* Sequential */ for (int k = 0; k < n; k++) {
			acc += alpha * A(i, k) * B(k, j);
		}
		C(i, j) *= beta + acc;
	}
}
",
lang: "cpp",
)

Lors de deux itérations (i, j, k) et (i, j, k+1), nous accédons à la même ligne de $A$ et la même colonne de $B$, ce qui est déjà mauvais pour la localité car $A$ a ses colonnes contiguës et $B$ a ses lignes contiguës.
Cependant, il ne semble pas y avoir de problème de faux partage, car $A$ et $B$ sont accédés en lecture seule, et il y a trop peu d'accès à $C$ pour que cela soit un problème.
Nous pouvons aussi observer le nombre de cache hit et cache miss pour ces 2 combinaisons pour différents nombre de threads:



SI + threads => + cache miss alors faux partage
SINON contention des threads

Maintenant que les combinaisons de Layouts mémoire problématiques ont été identifiées, j'ai décidé de ne tester que les 6 combinaisons restantes sur des matrices de taille $1000 * 1000$.
#figure(
	image(
		"results/strong_scaling_layout_minus_outliers.svg",
		width: 100%
	),
	caption: "Étude en strong scaling de l'impact des layouts mémoire sur les performances sur des matrices 1000x1000"
)

Les 2 meilleures combinaisons ont utilisé #emph[LayoutRight] pour $A$, #emph[LayoutLeft] pour $B$.
En effet, entre 2 itérations de la boucle la plus interne $k$, nous accédons à la même ligne de $A$ et la même colonne de $B$, donc avoir les lignes de $A$ contiguës et les colonnes de $B$ contiguës permet d'avoir un maximum de localité spatiale.

Nous pouvons aussi voir que les mesures sont similaires par paires, lorsque seul C change de layout.
$C$ étant accédée bien moins souvent ($i * j$ fois comparée à $i * j * k$ accès à $A$ et $B$) lors du calcul, il n'est pas surprenant de voir que son layout n'a pas d'impact considérable sur les performances.

= Autres optimisations

== Cache blocking

Le cache blocking est une optimisation classique qui consiste à changer l'ordre des itérations pour ré-utiliser les données dans le cache et améliorer la localité temporelle.

=== Mesure des cache hits et cache misses

=== Implémentation et vérification de l'algorithme

==== Implémentation 
Pour l'instruction la plus interne de notre kernel ```acc += alpha * A(i, k) * B(k, j);```, la colonne $j$ de $B$ est chargée en mémoire, il serait donc intéressant ici de la réutiliser pour les autres itérations de $i$ et $k$.
Donc, on va réordonner les itérations afin de faire quelques itérations sur $i$ à l'intérieur de la boucle $j$ afin de profiter du fait que la colonne $j$ de $B$ ait été chargée en mémoire.
#figure(
	grid(
		columns:2,
		code_snippet(code:"
for (int i = 0; i < int(A.extent(0)); i++) {
	for (int j = 0; j < int(B.extent(1)); j++) {
		double acc = 0.0;
		for (int k = 0; k < int(A.extent(1)); k++) {
			acc += alpha * A(i, k) * B(k, j);
		}
		C(i, j) *= beta + acc;
	}
}", lang: "cpp", title: "Sans cache blocking sur i")		,
		code_snippet(code:"
for (int i = 0; i < int(A.extent(0)); i += block_size) {
	for (int j = 0; j < int(B.extent(1)); j++) {
		for (int bi = i; bi < i + block_size; bi++) {
			double acc = 0.0;
			for (int k = 0; k < int(A.extent(1)); k++) {
				acc += alpha * A(bi, k) * B(k, j);
			}
			C(bi, j) *= beta + acc;
		}
	}
}", lang: "cpp", title: "Avec cache blocking sur i")
	)
, caption: "Kernel sans et avec cache blocking sur i")
Note: J'ai omit des détails comme la gestion de matrices dont la taille n'est pas divisible par la taille du block par soucis de clarté ici.

On peut réappliquer ce principe pour faire des portions d'itérations de boucles à l'intérieur les unes des autres, et avoir du cache blocking sur $i$, $j$, et $k$.

Cependant, pour $k$, le cache blocking est moins trivial.
En effet, pour chaque élément de la matrice $C$, nous devons faire attention à ce que l'accumulation est terminée avant de multiplier le tout.

Pour pouvoir faire des portions d'itérations $i$ et $j$ à l'intérieur de la boucle $k$, il faut créer une matrice temporaire contenant tous les accumulateurs pour tous les éléments des blocs des sous-itérations i et j.
De plus, il faut une matrice temporaire par thread sinon tous les résultats s'écrivent par dessus les autres.

Cela entraîne un coût mémoire supplémentaire, mais négligeable sachant que les matrices d'accumulateurs sont égales à la taille des blocs, qui sont très petits comparés aux matrices multipliées.

==== Vérification

Pour vérifier que notre algorithme est correct, nous allons le comparer à l'implémentation naïve.

Sachant que ni l'addition ni la multiplication flottante n'est associative et commutative, et donc que l'ordre des opérations a un impact sur le résultat,
nous allons vérifier que les 2 implémentations donnent un résultat identique à $#sym.epsilon = 10^(-9)$ près.

Après avoir généré aléatoirement 100 triplets de matrices avec des tailles aléatoires et des tailles de blocs aléatoires, aucun résultat n'est passé au-dessus de cette barre.
Ce n'est pas une preuve, mais l'addition étant (théoriquement) associative, il paraît naturel qu'on puisse faire l'accumulation dans le sens qu'on veut tant qu'on l'a fini avant de l'ajouter à l'élément de $C$ voulu.

=== Étude en strong scaling et tunning de la taille des blocs

Trouver la meilleur taille de blocs peut être fait de manière empirique, en testant différentes tailles de blocs et en mesurant le temps d'exécution.
Nous allons donc tester des tailles de blocs de 4, 8, 16, 32, 64, et 128, et mesurer le temps d'exécution pour chaque taille de block sur une matrice de taille $2000 * 2000$ (afin de mieux voir les gains), avec chaque niveau de blocking : sur $i$, sur $i j$, et sur $i j k$.
Le cache blocking sur $j$ uniquement étant inutile car on ne change pas l'ordre des itérations (i, j) 


J'ai vite remarqué que le cache blocking sur $i j k$ était aussi anormalement long comparé à celui sur $i$ et sur $i j$, donc je l'ai omis de l'étude.
Je pense que cela est dû au fait que k est la variable la plus interne, donc comme on travaille sur toute la ligne de $A$ et toute la colonne de $B$, faire une partie de l'itération $k$, changer de ligne pour $A$ ($i$) et de colonne de $B$ ($j$), puis revenir sur les anciens $i$ et $j$ est très coûteux.

Aussi, j'ai remarqué que les gains ne s'arrêtaient pas lorsqu'on augmente le nombre de threads au-delà de mon nombre de cœurs physiques (6). Donc ici, j'ai fait une étude jusqu'à 12 threads (6 cœurs physiques hyperthreadés).

#grid(
	columns: 2,
	rows: 1,
	row-gutter: 0.2cm,

	figure(
		image(
			"results/strong_scaling_cache_blocking_i.svg",
			width: 100%
		),
		caption: "Étude en strong scaling de l'impact du cache blocking sur i sur les performances. Le plus bas est mieux."
	)
	,
	figure(
		image(
			"results/strong_scaling_cache_blocking_ij.svg",
			width: 100%
		),
		caption: "Étude en strong scaling de l'impact du cache blocking sur i et j sur les performances. Le plus bas est mieux."
	)
)

Nous pouvons voir que le cache blocking sur $i$ apporte des gains de performances pour des blocs de taille 4 à 32, avec 8 étant la meilleure taille de bloc.
De plus, les courbes sont plus lisses, et donc plus prévisibles, ce qui est un bon point.
Pour 64, le cache-blocking semble n'avoir aucun avantage sur l'implémentation sans cache-blocking, et pour 128, il est même moins performant.

Pour le cache blocking sur $i$ et $j$, il y a un gain de performance quelque soit la taille de bloc, mais pas autant que pour le cache blocking sur $i$ pour les petites tailles de blocs.
Le cache blocking sur $i$ et $j$ semble donc plus robuste et général que celui sur $i$ uniquement, mais avec un gain de performance moins important.

D'après les tests, le cache blocking sur $i$ avec une taille de bloc de 8 semble être le meilleur choix. Mais peut-être que d'autres tailles de blocs seraient meilleures pour d'autres tailles de matrices, ou d'autres machines.

=== Analyse de l'amélioration de la localité temporelle

== Portage GPU

=== Motivation

=== Implémentation et vérification de l'algorithme

=== Impact sur les performances

Pour mesurer l'impact sur les performances, nous allons comparer les 2 implémentations, CPU avec 6 threads et cache-blocking tuné, et GPU, en augmentant la taille de la matrice.

Cependant, pour étudier les performances sur le GPU, il y a 2 benchmarks, l'un où l'on envoi les matrices avant de benchmarker, et l'autre où l'on envoi les matrices à chaque itération.
Cela permet de voir l'impact de la latence d'envoi des données sur les performances, et de voir à quel moment le surcoût d'envoi des données est compensé par le gain de performance du GPU.

= Conclusion

Dans ce TP, nous avons étudié et amélioré les performances de l'opération $C #sym.arrow.l #sym.alpha * A * B + #sym.beta * C$.
Pour cela, nous avons d'abord étudié l'impact des layouts mémoire sur les performances, puis nous avons implémenté et tuné un algorithme de cache blocking afin de maximiser la localité spatiale et temporelle.
Aussi, nous avons porté l'algorithme sur GPU pour le comparer à l'implémentation CPU.
Pour une matrice de taille $1500 * 1500$, nous sommes passés de XX à YY, où un speedup de ZZx a été observé.

Cependant, il est important de noter que les performances dépendent de la taille de la matrice, et que pour des matrices plus petites, l'implémentation CPU est plus performante.





#let appendix(body) = {
	set heading(numbering: "A", supplement: [Appendix])
	counter(heading).update(0)
	body
}

#show : appendix

= Machine et outils utilisés.

== Hardware

Informations obtenues grâce à lscpu :
- CPU AMD Ryzen 5 5500U, architecture x86_64, fréquence de base de 2100 MHz, turbo de 4056 MHz, capable de SIMD jusqu'à 256 bits (avx2, sse4.1)

Informations obtenues grâce à des sites externes de hardware #footnote[https://nanoreview.net/en/cpu/amd-ryzen-5-5500u] :
- GPU intégré Radeon Graphics RX Vega 7, fréquence de base de 300 MHz, turbo de 1800 MHz, 448 shading units, 7 execution units, capable de 1.6 TFLOP/s théoriquement.

Configuration obtenue grâce à lstopo :
```
Machine (7261MB total)
  Package L#0
    NUMANode L#0 (P#0 7261MB)
    L3 L#0 (4096KB)
      L2 L#0 (512KB) + L1d L#0 (32KB) + L1i L#0 (32KB) + Core L#0
        PU L#0 (P#0)
        PU L#1 (P#1)
      L2 L#1 (512KB) + L1d L#1 (32KB) + L1i L#1 (32KB) + Core L#1
        PU L#2 (P#2)
        PU L#3 (P#3)
      L2 L#2 (512KB) + L1d L#2 (32KB) + L1i L#2 (32KB) + Core L#2
        PU L#4 (P#4)
        PU L#5 (P#5)
    L3 L#1 (4096KB)
      L2 L#3 (512KB) + L1d L#3 (32KB) + L1i L#3 (32KB) + Core L#3
        PU L#6 (P#6)
        PU L#7 (P#7)
      L2 L#4 (512KB) + L1d L#4 (32KB) + L1i L#4 (32KB) + Core L#4
        PU L#8 (P#8)
        PU L#9 (P#9)
      L2 L#5 (512KB) + L1d L#5 (32KB) + L1i L#5 (32KB) + Core L#5
        PU L#10 (P#10)
        PU L#11 (P#11)
```

Ma machine est constituée de 6 cœurs physiques, chacun hyperthreadé pour un total de 12 cœurs logiques.

Aussi, ma machine possède 8Go de RAM, 2 caches L3 de taille 4096KB partagés entre triplets de cœurs physiques.
Chaque cœur physique possède 1 cache L2 de 512KB + 1 cache L1 d'instructions et 1 cache L1 de données tous deux de 32KB.

== Linux

- Distribution NixOS 24.05
- Kernel linux 6.6.68

== Software

- Compilateur: gcc 14.2.1
- OpenMP 4.5
- Kokkos 4.6.0 avec backend OpenMP
- nanobench 4.3.11
- perf 6.14.2
- lstopo 2.10.0
- lscpu de util-linux 2.39.4