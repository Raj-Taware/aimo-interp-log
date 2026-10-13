"""T-015 gate variants written by Claude (no big GPU for gpt-oss-120b; see
collab/plans/T-015-claude-run.md, "Kaggle-only"). One variant per (problem,
type), same five types and rules as stage2_variants.WRITER_PROMPT: answer
unchanged, no hints, LaTeX and $ delimiters kept.

    python scripts/label_gen/claude_variants_gate.py --problems <gate problems.jsonl> --out <dir>
writes <dir>/stage2_variants.jsonl in the stage-2 schema (writer = "claude").
TikZ figures (tv2-18/19) are carried over unchanged except under rename,
where point labels are renamed in the figure too.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import read_jsonl, write_jsonl

TIKZ_START = "\\begin{tikzpicture}"

V = {
"tv2-0": {
 "rephrase": r"""Suppose a real number $x$ obeys the equation $\sqrt[20]{x^{\log_{2026}x}}=26x$.
Multiply together every positive value of $x$ for which this holds. How many positive divisors does the resulting product have?""",
 "rename": r"""A real number $t$ satisfies $\sqrt[20]{t^{\log_{2026}t}}=26t$.
What is the number of positive divisors of the product of all possible positive values of $t$?""",
 "typos": r"""A real nubmer $x$ satisfies $\sqrt[20]{x^{\log_{2026}x}}=26x$.
What is the numebr of positive divisors of the prodcut of all possible postive values of $x$?""",
 "distract": r"""A real number $x$ satisfies $\sqrt[20]{x^{\log_{2026}x}}=26x$. This equation was first posed at a regional contest held in a hall with $412$ seats.
What is the number of positive divisors of the product of all possible positive values of $x$?""",
 "domain": r"""An engineer calibrating a sensor finds that its reading, a real number $x$, must satisfy $\sqrt[20]{x^{\log_{2026}x}}=26x$. She multiplies together all of the possible positive readings. How many positive divisors does her product have?""",
},
"tv2-1": {
 "rephrase": r"""Alice and Bob each hold an integer number of sweets, and each has a positive integer age. Alice observes: ``Adding each person's sweets to that person's age, my total is twice yours; multiplying each person's sweets by that person's age, my result is four times yours.'' Bob answers: ``If you handed me five of your sweets, our sums would be equal and so would our products.'' Find the product of Alice's age and Bob's age.""",
 "rename": r"""Priya and Omar are each holding some integer number of sweets. Priya says to Omar: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Omar replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Priya and Omar's ages?""",
 "typos": r"""Alice and Bob are each holding some integer numbre of sweets. Alice says to Bob: ``If we each added the number of sweets we're holidng to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yuors.'' Bob replies: ``Why don't you give me five of your sweets becuase then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?""",
 "distract": r"""Alice and Bob are each holding some integer number of sweets, which they bought from a shop that sells $12$ flavours. Alice says to Bob: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Bob replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?""",
 "domain": r"""Two warehouses, A and B, each hold some integer number of crates, and each has been open for a positive integer number of years. For each warehouse, add its crate count to its age in years: A's total is double B's. Multiply each warehouse's crate count by its age: A's result is four times B's. If A shipped five of its crates to B, then both the sums and the products for the two warehouses would become equal. What is the product of the two warehouses' ages?""",
},
"tv2-2": {
 "rephrase": r"""For some positive integer $n \geq 11$, Alice lists the integers $1, 2, \ldots, n$ on a board.
Bob deletes ten of these numbers, after which the average of the numbers left is $3000/37$.
Call the total of the ten deleted numbers $S$.
Find the remainder of $n \times S$ upon division by $997$.""",
 "rename": r"""Maya writes all positive integers from $1$ to $m$ on the board for some positive integer $m \geq 11$.
Leo then erases ten of them. The mean of the remaining numbers is $3000/37$.
The sum of the numbers Leo erased is $T$.
What is the remainder when $m \times T$ is divided by $997$?""",
 "typos": r"""Alice writes all positive integers from $1$ to $n$ on the borad for some positive integer $n \geq 11$.
Bob then erases ten of them. The mean of the remainign numbers is $3000/37$.
The sum of the numbers Bob erasd is $S$.
What is the remainder when $n \times S$ is dividid by $997$?""",
 "distract": r"""Alice writes all positive integers from $1$ to $n$ on the board for some positive integer $n \geq 11$. The board is $3$ meters wide.
Bob then erases ten of them. The mean of the remaining numbers is $3000/37$.
The sum of the numbers Bob erased is $S$.
What is the remainder when $n \times S$ is divided by $997$?""",
 "domain": r"""A raffle prints tickets numbered $1$ through $n$, for some positive integer $n \geq 11$. Ten of the tickets are lost, and the average number on the remaining tickets turns out to be $3000/37$. Let $S$ be the sum of the numbers on the ten lost tickets. What is the remainder when $n \times S$ is divided by $997$?""",
},
"tv2-3": {
 "rephrase": r"""A circle of radius $3$ is tangent to all four sides of an isosceles trapezoid whose area is $72$. The trapezoid's two parallel sides have unequal lengths $r$ and $s$. Compute $r^2+s^2$.""",
 "rename": r"""An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is $3$, and the area of the trapezoid is $72$. Let the parallel sides of the trapezoid have lengths $p$ and $q$, with $p \neq q$. Find $p^2+q^2$""",
 "typos": r"""An isoceles trapezoid has an inscribed circle tangent to each of its four sides. The raduis of the circle is $3$, and the area of the trapezoid is $72$. Let the paralel sides of the trapezoid have lengths $r$ and $s$, with $r \neq s$. Find $r^2+s^2$""",
 "distract": r"""An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is $3$, and the area of the trapezoid is $72$. The trapezoid is drawn on a sheet of paper that weighs $5$ grams. Let the parallel sides of the trapezoid have lengths $r$ and $s$, with $r \neq s$. Find $r^2+s^2$""",
 "domain": r"""A garden bed is shaped like an isosceles trapezoid, and a circular fountain of radius $3$ meters sits inside it, touching all four edges of the bed. The bed covers $72$ square meters. Its two parallel edges have different lengths $r$ and $s$ meters. Find $r^2+s^2$.""",
},
"tv2-4": {
 "rephrase": r"""There are $n$ marbles in an urn, each colored red or blue, with no fewer than $7$ of each color. Drawing $7$ marbles at random without replacement, the chance of getting exactly $4$ red ones is the same as the chance of getting exactly $5$ red ones. Determine the sum of the $5$ smallest possible values of $n$.""",
 "rename": r"""A jar contains $m$ beads. Each bead is either green or yellow, and there are at least $7$ beads of each color. When $7$ beads are drawn randomly from the jar without replacement, the probability that exactly $4$ of them are green equals the probability that exactly $5$ of them are green. Find the sum of the $5$ least values of $m$ for which this is possible.""",
 "typos": r"""An urn contains $n$ marbles. Each marble is either red or blue, and there are at least $7$ marbels of each color. When $7$ marbles are drawn randomly from the urn without replacment, the probabilty that exactly $4$ of them are red equals the probability that exactly $5$ of them are red. Find the sum of the $5$ least values of $n$ for which this is possibel.""",
 "distract": r"""An urn contains $n$ marbles. Each marble is either red or blue, and there are at least $7$ marbles of each color. Each marble has a diameter of $2$ centimeters. When $7$ marbles are drawn randomly from the urn without replacement, the probability that exactly $4$ of them are red equals the probability that exactly $5$ of them are red. Find the sum of the $5$ least values of $n$ for which this is possible.""",
 "domain": r"""A committee of $7$ is chosen uniformly at random from a club of $n$ members, each of whom is either a student or a teacher, with at least $7$ students and at least $7$ teachers. The probability that the committee contains exactly $4$ students equals the probability that it contains exactly $5$ students. Find the sum of the $5$ least values of $n$ for which this is possible.""",
},
"tv2-5": {
 "rephrase": r"""For each positive integer $n$, let
\begin{equation*}
    f(n) = \sum_{i = 1}^n \sum_{j = 1}^n j^{3272} \left\lfloor\frac1j + \frac{n-i}{n}\right\rfloor,
\end{equation*}
so that $f$ maps positive integers to positive integers. Put $M=2 \cdot 3 \cdot 5 \cdot 7 \cdot 11 \cdot 13$ and $N = f{\left(M^{7}\right)} - f{\left(M^{7}-1\right)}$. If $2^k$ is the largest power of $2$ dividing $N$, with $k$ a non-negative integer, find the remainder of $2^k$ modulo $20^3$.""",
 "rename": r"""Define a function $g \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ by
\begin{equation*}
    g(m) = \sum_{a = 1}^m \sum_{b = 1}^m b^{3272} \left\lfloor\frac1b + \frac{m-a}{m}\right\rfloor.
\end{equation*}
Let $P=2 \cdot 3 \cdot 5 \cdot 7 \cdot 11 \cdot 13$ and let $Q = g{\left(P^{7}\right)} - g{\left(P^{7}-1\right)}$. Let $t$ be the largest non-negative integer such that $2^t$ divides $Q$. What is the remainder when $2^t$ is divided by $20^3$?""",
 "typos": r"""Define a funtion $f \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ by
\begin{equation*}
    f(n) = \sum_{i = 1}^n \sum_{j = 1}^n j^{3272} \left\lfloor\frac1j + \frac{n-i}{n}\right\rfloor.
\end{equation*}
Let $M=2 \cdot 3 \cdot 5 \cdot 7 \cdot 11 \cdot 13$ and let $N = f{\left(M^{7}\right)} - f{\left(M^{7}-1\right)}$. Let $k$ be the largset non-negative integer such that $2^k$ divdes $N$. What is the remainder when $2^k$ is divided by $20^3$?""",
 "distract": r"""Define a function $f \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ by
\begin{equation*}
    f(n) = \sum_{i = 1}^n \sum_{j = 1}^n j^{3272} \left\lfloor\frac1j + \frac{n-i}{n}\right\rfloor.
\end{equation*}
Let $M=2 \cdot 3 \cdot 5 \cdot 7 \cdot 11 \cdot 13$ and let $N = f{\left(M^{7}\right)} - f{\left(M^{7}-1\right)}$. Note that $M$ is also the number of seconds in a little over $8$ hours. Let $k$ be the largest non-negative integer such that $2^k$ divides $N$. What is the remainder when $2^k$ is divided by $20^3$?""",
 "domain": r"""A bookkeeping program assigns a positive integer weight to each positive integer $n$ by
\begin{equation*}
    f(n) = \sum_{i = 1}^n \sum_{j = 1}^n j^{3272} \left\lfloor\frac1j + \frac{n-i}{n}\right\rfloor.
\end{equation*}
The program's batch size is $M^{7}$, where $M=2 \cdot 3 \cdot 5 \cdot 7 \cdot 11 \cdot 13$, and the auditor looks at the increase $N = f{\left(M^{7}\right)} - f{\left(M^{7}-1\right)}$. Let $k$ be the largest non-negative integer such that $2^k$ divides $N$. What is the remainder when $2^k$ is divided by $20^3$?""",
},
"tv2-6": {
 "rephrase": r"""How many integers not exceeding 100 can be written in the form $a+b+ab$, where a and b are two different positive integers?""",
 "rename": r"""Find the number of integers less than or equal to 100 that are equal to $p+q+pq$ for some choice of distinct positive integers p and q.""",
 "typos": r"""Find the numbr of integers less than or equal to 100 that are equal to $a+b+ab$ for some choise of distinct postive integers a and b.""",
 "distract": r"""Find the number of integers less than or equal to 100 that are equal to $a+b+ab$ for some choice of distinct positive integers a and b. (The number 100 is also the boiling point of water in degrees Celsius.)""",
 "domain": r"""A shop sells a bundle whose price, in dollars, is $a+b+ab$, where $a$ and $b$ are the numbers of two different kinds of items in the bundle, each a positive integer and with $a \neq b$. How many whole-dollar prices of at most 100 dollars can such a bundle have?""",
},
"tv2-7": {
 "rephrase": r"""Over all integer bases $b>9$ such that $97_b$ is divisible by $17_b$, what is the sum of the bases $b$?""",
 "rename": r"""Find the sum of all integer bases $k>9$ for which $17_k$ is a divisor of $97_k.$""",
 "typos": r"""Find the sum of all integr bases $b>9$ for wich $17_b$ is a divsor of $97_b.$""",
 "distract": r"""Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$ In base ten, the number $97$ is the largest two-digit prime.""",
 "domain": r"""An alien civilization writes numbers in an integer base $b>9$. In their notation, a crate holds $97_b$ apples, and the apples can be split evenly into bags of $17_b$ apples each. Find the sum of all possible values of $b$.""",
},
"tv2-8": {
 "rephrase": r"""Consider the parabola $y = \frac{1}{2}(x-1)^2 - 6$ and circles centered at $(1, 79)$. Add up all real numbers $r$ for which the circle of radius $r$ with that center is tangent to the parabola at one or more points.""",
 "rename": r"""Find the sum of all real numbers $\rho$ such that there is at least one point where the circle with radius $\rho$ centered at $(1, 79)$ is tangent to the parabola with equation $v = \frac{1}{2}(u-1)^2 - 6.$""",
 "typos": r"""Find the sum of all real numbers $r$ such that there is at least one piont where the circle with radius $r$ centred at $(1, 79)$ is tangent to the parabola with equaton $y = \frac{1}{2}(x-1)^2 - 6.$""",
 "distract": r"""Find the sum of all real numbers $r$ such that there is at least one point where the circle with radius $r$ centered at $(1, 79)$ is tangent to the parabola with equation $y = \frac{1}{2}(x-1)^2 - 6.$ The parabola's name comes from a Greek word, and the figure was first studied over $2000$ years ago.""",
 "domain": r"""A satellite dish has a cross-section shaped like the curve $y = \frac{1}{2}(x-1)^2 - 6$ in a coordinate system. A circular ring centered at the point $(1, 79)$ is to touch this curve tangentially at least once. Find the sum of all real values of the ring's radius $r$ for which this can happen.""",
},
"tv2-9": {
 "rephrase": r"""Consider the parabola $2y = x^2 - 8x + 12$ and circles centered at $(4, 39)$. Add up all real numbers $r$ for which the circle of radius $r$ with that center is tangent to the parabola at one or more points.""",
 "rename": r"""Find the sum of all real numbers $\rho$ such that there is at least one point where the circle with radius $\rho$ centered at $(4, 39)$ is tangent to the parabola with equation $2v = u^2 - 8u + 12.$""",
 "typos": r"""Find the sum of all real numbers $r$ such that there is at least one piont where the circle with radius $r$ centred at $(4, 39)$ is tangent to the parabola with equaton $2y = x^2 - 8x + 12.$""",
 "distract": r"""Find the sum of all real numbers $r$ such that there is at least one point where the circle with radius $r$ centered at $(4, 39)$ is tangent to the parabola with equation $2y = x^2 - 8x + 12.$ The parabola's name comes from a Greek word, and the figure was first studied over $2000$ years ago.""",
 "domain": r"""A satellite dish has a cross-section shaped like the curve $2y = x^2 - 8x + 12$ in a coordinate system. A circular ring centered at the point $(4, 39)$ is to touch this curve tangentially at least once. Find the sum of all real values of the ring's radius $r$ for which this can happen.""",
},
"tv2-10": {
 "rephrase": r"""Consider every arithmetic sequence of integers whose first term is $4$ and which contains both $24$ and $34$ among its terms. Add up the $10$th terms of all such sequences.""",
 "rename": r"""Find the sum of the terms at position $10$ of all arithmetic progressions of integers that have initial term equal to $4$ and include both $24$ and $34$ as terms.""",
 "typos": r"""Find the sum of the terms at positon $10$ of all arithmetic sequences of integers that have frist term equal to $4$ and inculde both $24$ and $34$ as terms.""",
 "distract": r"""Find the sum of the terms at position $10$ of all arithmetic sequences of integers that have first term equal to $4$ and include both $24$ and $34$ as terms. Arithmetic sequences were studied by the ancient Babylonians about $4000$ years ago.""",
 "domain": r"""A savings plan starts with $4$ dollars in the account, and every month the balance changes by the same fixed integer number of dollars. At some months the balance is exactly $24$ dollars and at some months it is exactly $34$ dollars. Counting the starting balance as month $1$, find the sum of the balances in month $10$ over all possible such plans.""",
},
"tv2-11": {
 "rephrase": r"""In triangle $\triangle ABC$, sides $AB$ and $BC$ are equal, and $I$ denotes the incenter. The ratio of the perimeter of $\triangle ABC$ to that of $\triangle AIC$ is $125:6,$ and every side of each of the two triangles is an integer. What is the smallest possible value of $AB$?""",
 "rename": r"""Isosceles triangle $\triangle PQR$ has $PQ = QR.$ Let $J$ be the incenter of $\triangle PQR.$ The perimeters of $\triangle PQR$ and $\triangle PJR$ are in the ratio $125:6,$ and all the sides of both triangles have integer lengths. Find the minimum possible value of $PQ.$""",
 "typos": r"""Isoceles triangle $\triangle ABC$ has $AB = BC.$ Let $I$ be the incenter of $\triangle ABC.$ The perimiters of $\triangle ABC$ and $\triangle AIC$ are in the ratio $125:6,$ and all the sides of both triangels have integer lengths. Find the minimun possible value of $AB.$""",
 "distract": r"""Isosceles triangle $\triangle ABC$ has $AB = BC.$ Let $I$ be the incenter of $\triangle ABC.$ The circumcenter of $\triangle ABC$ is labelled $O$. The perimeters of $\triangle ABC$ and $\triangle AIC$ are in the ratio $125:6,$ and all the sides of both triangles have integer lengths. Find the minimum possible value of $AB.$""",
 "domain": r"""A surveyor marks three posts $A$, $B$, $C$ forming a triangular field with $AB = BC$, and places a well at the point $I$ equidistant from all three sides of the field. The fence around field $ABC$ and the fence around the smaller triangular plot $AIC$ have lengths in the ratio $125:6,$ and every side of both triangles is a whole number of meters. Find the minimum possible length of $AB$ in meters.""",
},
"tv2-12": {
 "rephrase": r"""How many positive divisors of $17017^{17}$ have remainder $5$ upon division by $12$? Call this count $N$ and find $N$.""",
 "rename": r"""Let $K$ be the number of positive integer divisors of $17017^{17}$ that leave a remainder of $5$ when divided by $12$. Find $K$.""",
 "typos": r"""Let $N$ be the number of postive integer divisors of $17017^{17}$ that leave a remainer of $5$ when divided by $12$. Find $N$.""",
 "distract": r"""Let $N$ be the number of positive integer divisors of $17017^{17}$ that leave a remainder of $5$ when divided by $12$. A year has $12$ months. Find $N$.""",
 "domain": r"""A warehouse holds $17017^{17}$ identical boxes, and the manager wants to arrange them in equal stacks. Call a stack size usable if it divides $17017^{17}$ and leaves a remainder of $5$ when divided by $12$. Let $N$ be the number of usable stack sizes. Find $N$.""",
},
"tv2-13": {
 "rephrase": r"""Define $S$ as the sum of the infinite series
\[
\frac{1}{9} + \frac{1}{99} + \frac{1}{999} + \frac{1}{9999}  + \cdots
\]
What is the remainder when $\lfloor 10^{100} S \rfloor$ is divided by $1000$?""",
 "rename": r"""Let $T$ denote the value of the infinite sum
\[
\frac{1}{9} + \frac{1}{99} + \frac{1}{999} + \frac{1}{9999}  + \cdots
\]
Find the remainder when the greatest integer less than or equal to $10^{100} T$ is divided by $1000.$""",
 "typos": r"""Let $S$ denote the value of the infinte sum
\[
\frac{1}{9} + \frac{1}{99} + \frac{1}{999} + \frac{1}{9999}  + \cdots
\]
Find the remiander when the greatest integer less than or equal to $10^{100} S$ is divded by $1000.$""",
 "distract": r"""Let $S$ denote the value of the infinite sum
\[
\frac{1}{9} + \frac{1}{99} + \frac{1}{999} + \frac{1}{9999}  + \cdots
\]
The number $10^{100}$ is called a googol. Find the remainder when the greatest integer less than or equal to $10^{100} S$ is divided by $1000.$""",
 "domain": r"""A machine pours liquid into a tank in stages: in stage $1$ it adds $\frac{1}{9}$ liters, in stage $2$ it adds $\frac{1}{99}$ liters, in stage $3$ it adds $\frac{1}{999}$ liters, and so on forever, in stage $k$ adding $\frac{1}{10^k-1}$ liters. Let $S$ be the total volume in liters. Find the remainder when the greatest integer less than or equal to $10^{100} S$ is divided by $1000.$""",
},
"tv2-14": {
 "rephrase": r"""In $\triangle ABC$, the bisector of $\angle BAC$ meets $\overline{BC}$ at $D.$ A circle $\omega$ goes through $A$ and touches segment $\overline{BC}$ at $D.$ It meets segments $\overline{AB}$ and $\overline{AC}$ again at $E \neq A$ and $F \neq A,$ respectively. Given $AB = 1628$ and $AC = 932,$ and that $AE, AF, BD,$ and $CD$ are all positive integers, find the sum of all possible values of $BC.$""",
 "rename": r"""Let $\triangle PQR$ be a triangle with $S$ on $\overline{QR}$ such that $\overline{PS}$ bisects $\angle QPR.$ Let $\Gamma$ be the circle that passes through $P$ and is tangent to segment $\overline{QR}$ at $S.$ Let $U \neq P$ and $W \neq P$ be the intersections of $\Gamma$ with segments $\overline{PQ}$ and $\overline{PR},$ respectively. Suppose that $PQ = 1628, PR = 932,$ and all of $PU, PW, QS,$ and $RS$ are positive integers. Find the sum of all possible values of $QR.$""",
 "typos": r"""Let $\triangle ABC$ be a triangle with $D$ on $\overline{BC}$ such that $\overline{AD}$ bisects $\angle BAC.$ Let $\omega$ be the cirlce that passes through $A$ and is tangent to segment $\overline{BC}$ at $D.$ Let $E \neq A$ and $F \neq A$ be the intersectons of $\omega$ with segments $\overline{AB}$ and $\overline{AC},$ respectivly. Suppose that $AB = 1628, AC = 932,$ and all of $AE, AF, BD,$ and $CD$ are positive integres. Find the sum of all possible values of $BC.$""",
 "distract": r"""Let $\triangle ABC$ be a triangle with $D$ on $\overline{BC}$ such that $\overline{AD}$ bisects $\angle BAC.$ Let $\omega$ be the circle that passes through $A$ and is tangent to segment $\overline{BC}$ at $D.$ Let $E \neq A$ and $F \neq A$ be the intersections of $\omega$ with segments $\overline{AB}$ and $\overline{AC},$ respectively. Let $H$ be the orthocenter of $\triangle ABC.$ Suppose that $AB = 1628, AC = 932,$ and all of $AE, AF, BD,$ and $CD$ are positive integers. Find the sum of all possible values of $BC.$""",
 "domain": r"""Three towns $A$, $B$, $C$ are joined by straight roads, with $AB = 1628$ meters and $AC = 932$ meters. A path from $A$ bisects the angle at $A$ and meets road $BC$ at point $D.$ A circular cycling track passes through $A$ and just touches road $BC$ at $D$; it crosses road $AB$ again at $E \neq A$ and road $AC$ again at $F \neq A.$ The distances $AE, AF, BD,$ and $CD$ are all positive whole numbers of meters. Find the sum of all possible lengths of road $BC$ in meters.""",
},
"tv2-15": {
 "rephrase": r"""In $\triangle ABC$, the bisector of $\angle BAC$ meets $\overline{BC}$ at $D.$ A circle $\omega$ goes through $A$ and touches segment $\overline{BC}$ at $D.$ It meets segments $\overline{AB}$ and $\overline{AC}$ again at $E \neq A$ and $F \neq A,$ respectively. Given $AB = 200$ and $AC = 225,$ and that $AE, AF, BD,$ and $CD$ are all positive integers, find the sum of all possible values of $BC.$""",
 "rename": r"""Let $\triangle PQR$ be a triangle with $S$ on $\overline{QR}$ such that $\overline{PS}$ bisects $\angle QPR.$ Let $\Gamma$ be the circle that passes through $P$ and is tangent to segment $\overline{QR}$ at $S.$ Let $U \neq P$ and $W \neq P$ be the intersections of $\Gamma$ with segments $\overline{PQ}$ and $\overline{PR},$ respectively. Suppose that $PQ = 200, PR = 225,$ and all of $PU, PW, QS,$ and $RS$ are positive integers. Find the sum of all possible values of $QR.$""",
 "typos": r"""Let $\triangle ABC$ be a triangle with $D$ on $\overline{BC}$ such that $\overline{AD}$ bisects $\angle BAC.$ Let $\omega$ be the cirlce that passes through $A$ and is tangent to segment $\overline{BC}$ at $D.$ Let $E \neq A$ and $F \neq A$ be the intersectons of $\omega$ with segments $\overline{AB}$ and $\overline{AC},$ respectivly. Suppose that $AB = 200, AC = 225,$ and all of $AE, AF, BD,$ and $CD$ are positive integres. Find the sum of all possible values of $BC.$""",
 "distract": r"""Let $\triangle ABC$ be a triangle with $D$ on $\overline{BC}$ such that $\overline{AD}$ bisects $\angle BAC.$ Let $\omega$ be the circle that passes through $A$ and is tangent to segment $\overline{BC}$ at $D.$ Let $E \neq A$ and $F \neq A$ be the intersections of $\omega$ with segments $\overline{AB}$ and $\overline{AC},$ respectively. Let $H$ be the orthocenter of $\triangle ABC.$ Suppose that $AB = 200, AC = 225,$ and all of $AE, AF, BD,$ and $CD$ are positive integers. Find the sum of all possible values of $BC.$""",
 "domain": r"""Three towns $A$, $B$, $C$ are joined by straight roads, with $AB = 200$ meters and $AC = 225$ meters. A path from $A$ bisects the angle at $A$ and meets road $BC$ at point $D.$ A circular cycling track passes through $A$ and just touches road $BC$ at $D$; it crosses road $AB$ again at $E \neq A$ and road $AC$ again at $F \neq A.$ The distances $AE, AF, BD,$ and $CD$ are all positive whole numbers of meters. Find the sum of all possible lengths of road $BC$ in meters.""",
},
"tv2-16": {
 "rephrase": r"""Consider functions $f \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ satisfying
\begin{equation*}
    f(m) + f(n) = f(m + n + mn)
\end{equation*}
for every pair of positive integers $m$ and $n$, and also $f(n) \leq 617$ whenever $n \leq 617$. Over all such functions, how many distinct values are possible for $f(3520)$?""",
 "rename": r"""Let $g \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ be a function such that for all positive integers $a$ and $b$,
\begin{equation*}
    g(a) + g(b) = g(a + b + ab).
\end{equation*}
Across all functions $g$ such that $g(a) \leq 617$ for all $a \leq 617$, how many different values can $g(3520)$ take?""",
 "typos": r"""Let $f \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ be a funtion such that for all positive integers $m$ and $n$,
\begin{equation*}
    f(m) + f(n) = f(m + n + mn).
\end{equation*}
Acros all functions $f$ such that $f(n) \leq 617$ for all $n \leq 617$, how many diferent values can $f(3520)$ take?""",
 "distract": r"""Let $f \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ be a function such that for all positive integers $m$ and $n$,
\begin{equation*}
    f(m) + f(n) = f(m + n + mn).
\end{equation*}
Functional equations of this kind are popular in olympiads held in over $100$ countries. Across all functions $f$ such that $f(n) \leq 617$ for all $n \leq 617$, how many different values can $f(3520)$ take?""",
 "domain": r"""A scoring system gives every positive integer $n$ a positive integer score $f(n)$. The rules guarantee that for all positive integers $m$ and $n$, the scores satisfy
\begin{equation*}
    f(m) + f(n) = f(m + n + mn),
\end{equation*}
and no positive integer $n \leq 617$ receives a score above $617$. Considering every scoring system that follows these rules, how many different scores can the number $3520$ receive?""",
},
"tv2-17": {
 "rephrase": r"""For a real number $k$, the system
\begin{align*} |25+20i-z|&=5\\ |z-4-k|&=|z-3i-k| \\ \end{align*}
has exactly one complex solution $z$. Writing the sum of all such $k$ as $\frac{m}{n}$ in lowest terms with $m$ and $n$ positive integers, compute $m+n.$ (Here $i=\sqrt{-1}.$)""",
 "rename": r"""Let $c$ be a real number such that the system
\begin{align*} |25+20i-w|&=5\\ |w-4-c|&=|w-3i-c| \\ \end{align*}
has exactly one complex solution $w$. The sum of all possible values of $c$ can be written as $\frac{p}{q},$ where $p$ and $q$ are relatively prime positive integers. Find $p+q.$ Here $i=\sqrt{-1}.$""",
 "typos": r"""Let $k$ be a real number such that the sytem
\begin{align*} |25+20i-z|&=5\\ |z-4-k|&=|z-3i-k| \\ \end{align*}
has exactly one complex soluton $z$. The sum of all posible values of $k$ can be written as $\frac{m}{n},$ where $m$ and $n$ are relatively prime positive integers. Find $m+n.$ Here $i=\sqrt{-1}.$""",
 "distract": r"""Let $k$ be a real number such that the system
\begin{align*} |25+20i-z|&=5\\ |z-4-k|&=|z-3i-k| \\ \end{align*}
has exactly one complex solution $z$. The complex numbers were first used systematically in the $16$th century. The sum of all possible values of $k$ can be written as $\frac{m}{n},$ where $m$ and $n$ are relatively prime positive integers. Find $m+n.$ Here $i=\sqrt{-1}.$""",
 "domain": r"""On a map drawn on the complex plane, a radio tower at the point $25+20i$ reaches exactly the points at distance $5$ from it. For a real number $k$, a road consists of all points $z$ that are equally far from $4+k$ and from $3i+k$. For certain values of $k$, the road meets the tower's boundary circle at exactly one point. The sum of all such values of $k$ can be written as $\frac{m}{n},$ where $m$ and $n$ are relatively prime positive integers. Find $m+n.$ Here $i=\sqrt{-1}.$""",
},
"tv2-18": {
 "rephrase": r"""In $\triangle ABC$, the points $A, D, E, B$ appear in that order along side $\overline{AB}$, where $AD = 14$, $DE = 44$, and $EB = 30$. Along side $\overline{AC}$ the points $A, F, G, C$ appear in that order, where $AF = 9$, $FG = 144$, and $GC = 255$. Reflect $D$ through $F$ to get $M$, and reflect $G$ through $E$ to get $N$. The area of quadrilateral $DEGF$ is $243$. What is the area of heptagon $AFNBCEM$?
""",
 "rename": r"""On $\triangle PQR$ points $P, S, T$, and $Q$ lie in that order on side $\overline{PQ}$ with $PS = 14$, $ST = 44$, $TQ = 30$. Points $P, U, W$ and $R$ lie in that order on side $\overline{PR}$ with $PU = 9$, $UW = 144$, and $WR = 255$. Let $X$ be the reflection of $S$ through $U$, and let $Y$ be the reflection of $W$ through $T$. Quadrilateral $STWU$ has area $243$. Find the area of heptagon $PUYQRTX$.
""",
 "typos": r"""On $\triangle ABC$ points $A, D, E$, and $B$ lie in that ordre on side $\overline{AB}$ with $AD = 14$, $DE = 44$, $EB = 30$. Points $A, F, G$ and $C$ lie in that order on side $\overline{AC}$ with $AF = 9$, $FG = 144$, and $GC = 255$. Let $M$ be the reflction of $D$ through $F$, and let $N$ be the reflection of $G$ thorugh $E$. Quadrilateral $DEGF$ has area $243$. Find the area of heptagon $AFNBCEM$.
""",
 "distract": r"""On $\triangle ABC$ points $A, D, E$, and $B$ lie in that order on side $\overline{AB}$ with $AD = 14$, $DE = 44$, $EB = 30$. Points $A, F, G$ and $C$ lie in that order on side $\overline{AC}$ with $AF = 9$, $FG = 144$, and $GC = 255$. The figure was drawn with a pencil $18$ centimeters long. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area $243$. Find the area of heptagon $AFNBCEM$.
""",
 "domain": r"""A triangular park has corners $A$, $B$, $C$. Along the straight edge from $A$ to $B$, a gardener plants markers $D$ and $E$ so that $A, D, E, B$ come in that order, with $AD = 14$, $DE = 44$, and $EB = 30$ meters. Along the edge from $A$ to $C$, markers $F$ and $G$ are placed so that $A, F, G, C$ come in that order, with $AF = 9$, $FG = 144$, and $GC = 255$ meters. Lamp $M$ is placed at the reflection of $D$ through $F$, and lamp $N$ at the reflection of $G$ through $E$. The quadrilateral flower bed $DEGF$ covers $243$ square meters. Find the area, in square meters, of the heptagonal region $AFNBCEM$.
""",
},
"tv2-19": {
 "rephrase": r"""In $\triangle ABC$, the points $A, D, E, B$ appear in that order along side $\overline{AB}$, where $AD = 4$, $DE = 16$, and $EB = 8$. Along side $\overline{AC}$ the points $A, F, G, C$ appear in that order, where $AF = 13$, $FG = 52$, and $GC = 26$. Reflect $D$ through $F$ to get $M$, and reflect $G$ through $E$ to get $N$. The area of quadrilateral $DEGF$ is $288$. What is the area of heptagon $AFNBCEM$?
""",
 "rename": r"""On $\triangle PQR$ points $P, S, T$, and $Q$ lie in that order on side $\overline{PQ}$ with $PS = 4$, $ST = 16$, $TQ = 8$. Points $P, U, W$ and $R$ lie in that order on side $\overline{PR}$ with $PU = 13$, $UW = 52$, and $WR = 26$. Let $X$ be the reflection of $S$ through $U$, and let $Y$ be the reflection of $W$ through $T$. Quadrilateral $STWU$ has area $288$. Find the area of heptagon $PUYQRTX$.
""",
 "typos": r"""On $\triangle ABC$ points $A, D, E$, and $B$ lie in that ordre on side $\overline{AB}$ with $AD = 4$, $DE = 16$, $EB = 8$. Points $A, F, G$ and $C$ lie in that order on side $\overline{AC}$ with $AF = 13$, $FG = 52$, and $GC = 26$. Let $M$ be the reflction of $D$ through $F$, and let $N$ be the reflection of $G$ thorugh $E$. Quadrilateral $DEGF$ has area $288$. Find the area of heptagon $AFNBCEM$.
""",
 "distract": r"""On $\triangle ABC$ points $A, D, E$, and $B$ lie in that order on side $\overline{AB}$ with $AD = 4$, $DE = 16$, $EB = 8$. Points $A, F, G$ and $C$ lie in that order on side $\overline{AC}$ with $AF = 13$, $FG = 52$, and $GC = 26$. The figure was drawn with a pencil $18$ centimeters long. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area $288$. Find the area of heptagon $AFNBCEM$.
""",
 "domain": r"""A triangular park has corners $A$, $B$, $C$. Along the straight edge from $A$ to $B$, a gardener plants markers $D$ and $E$ so that $A, D, E, B$ come in that order, with $AD = 4$, $DE = 16$, and $EB = 8$ meters. Along the edge from $A$ to $C$, markers $F$ and $G$ are placed so that $A, F, G, C$ come in that order, with $AF = 13$, $FG = 52$, and $GC = 26$ meters. Lamp $M$ is placed at the reflection of $D$ through $F$, and lamp $N$ at the reflection of $G$ through $E$. The quadrilateral flower bed $DEGF$ covers $288$ square meters. Find the area, in square meters, of the heptagonal region $AFNBCEM$.
""",
},
"tv2-20": {
 "rephrase": r"""Patrick set off from school toward the park, walking along a straight road at a steady pace.
Tanya left the school one hour after Patrick, running the same route at a steady pace that was $2$ miles per hour faster than Patrick's.
Jose left one hour after Tanya, bicycling the same route at a steady speed $7$ miles per hour faster than Tanya's.
The three of them reached the park simultaneously.
If the school-to-park distance is $\frac{m}{n}$ miles with $m$ and $n$ relatively prime positive integers, find $m + n$.""",
 "rename": r"""Omar started walking at a constant rate along a straight road from school to the park.
One hour after Omar left, Priya started running along the same road from school to the park.
One hour after Priya left, Lena started bicycling along the same road from school to the park.
Priya ran at a constant rate of $2$ miles per hour faster than Omar walked,
Lena bicycled at a constant rate of $7$ miles per hour faster than Priya ran,
and all three arrived at the park at the same time.
The distance from the school to the park is $\frac{p}{q}$ miles,
where $p$ and $q$ are relatively prime positive integers. Find $p + q$.""",
 "typos": r"""Patrick started walking at a constant rate along a straight raod from school to the park.
One hour after Patrick left, Tanya started running along the same road from school to the park.
One hour after Tanya left, Jose started bicylcing along the same road from school to the park.
Tanya ran at a constant rate of $2$ miles per hour faster than Patrick walked,
Jose bicycled at a constant rate of $7$ miles per hour faster than Tanya ran,
and all three arived at the park at the same tiem.
The distance from the school to the park is $\frac{m}{n}$ miles,
where $m$ and $n$ are relatively prime positive integers. Find $m + n$.""",
 "distract": r"""Patrick started walking at a constant rate along a straight road from school to the park.
One hour after Patrick left, Tanya started running along the same road from school to the park.
One hour after Tanya left, Jose started bicycling along the same road from school to the park.
Patrick was carrying a backpack that weighed $6$ pounds.
Tanya ran at a constant rate of $2$ miles per hour faster than Patrick walked,
Jose bicycled at a constant rate of $7$ miles per hour faster than Tanya ran,
and all three arrived at the park at the same time.
The distance from the school to the park is $\frac{m}{n}$ miles,
where $m$ and $n$ are relatively prime positive integers. Find $m + n$.""",
 "domain": r"""Three delivery drones leave the same depot and fly along the same straight route to a customer.
The first drone flies at a constant speed. One hour after it departs, the second drone departs, flying at a constant speed $2$ miles per hour faster than the first.
One hour after the second departs, the third departs, flying at a constant speed $7$ miles per hour faster than the second.
All three drones reach the customer at the same moment.
The distance from the depot to the customer is $\frac{m}{n}$ miles,
where $m$ and $n$ are relatively prime positive integers. Find $m + n$.""",
},
"tv2-21": {
 "rephrase": r"""Let $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for every $n \geq 1$ define the Fibonacci numbers.
Let $N$ count the positive integers $n < 10^{101}$ for which $5$ divides $n^2 + (n+1)^2$ while $5$ does not divide $F_{n-1}^2 + F_n^2$. Counting multiplicity, how many prime factors does $N$ have?""",
 "rename": r"""The Fibonacci numbers are defined as follows: $G_0 = 0$, $G_1 = 1$, and $G_{m+1} = G_m + G_{m-1}$ for $m \geq 1$.
There are $K$ positive integers $m$ strictly less than $10^{101}$ such that $m^2 + (m+1)^2$ is a multiple of 5
but $G_{m-1}^2 + G_m^2$ is not. How many prime factors does $K$ have, counted with multiplicity?""",
 "typos": r"""The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$.
There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multple of 5
but $F_{n-1}^2 + F_n^2$ is not. How many prime factros does $N$ have, counted with multiplicty?""",
 "distract": r"""The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$.
Leonardo of Pisa described them in a book published in the year $1202$.
There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5
but $F_{n-1}^2 + F_n^2$ is not. How many prime factors does $N$ have, counted with multiplicity?""",
 "domain": r"""A rabbit colony grows by the Fibonacci rule: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$, where $F_n$ is the number of pairs in month $n$.
A researcher counts the months $n$, with $n$ a positive integer strictly less than $10^{101}$, for which $n^2 + (n+1)^2$ is a multiple of 5
but $F_{n-1}^2 + F_n^2$ is not. Let $N$ be the number of such months. How many prime factors does $N$ have, counted with multiplicity?""",
},
"tv2-22": {
 "rephrase": r"""In triangle $ABC$, side $AB$ has length $20$ and the circumscribed circle has radius $R = 26$. Drop a perpendicular from $C$ to line $AB$ with foot $D$. Find the largest possible length of $CD$.""",
 "rename": r"""Triangle $PQR$ has side length $PQ = 20$ and circumradius $\rho = 26$. Let $S$ be the foot of the perpendicular from $R$ to the line $PQ$. What is the greatest possible length of segment $RS$?""",
 "typos": r"""Triangle $ABC$ has side lenght $AB = 20$ and circumradius $R = 26$. Let $D$ be the foot of the perpendiculr from $C$ to the line $AB$. What is the greatst possible length of segment $CD$?""",
 "distract": r"""Triangle $ABC$ has side length $AB = 20$ and circumradius $R = 26$. The triangle's centroid is labelled $G$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?""",
 "domain": r"""Three lighthouses $A$, $B$, $C$ all stand on a circular coastline of radius $26$ kilometers, and lighthouses $A$ and $B$ are $20$ kilometers apart in a straight line. A boat sails from $C$ straight to the line through $A$ and $B$, meeting it at a right angle at point $D$. What is the greatest possible length of the trip $CD$ in kilometers?""",
},
}

RENAME_TIKZ = {"A": "P", "B": "Q", "C": "R", "D": "S", "E": "T", "F": "U", "G": "W", "M": "X", "N": "Y"}


def rename_tikz(tikz: str) -> str:
    tikz = re.sub(r"\((\w)\)", lambda m: f"({RENAME_TIKZ.get(m.group(1), m.group(1))})", tikz)
    tikz = re.sub(r"\$(\w)\$", lambda m: f"${RENAME_TIKZ.get(m.group(1), m.group(1))}$", tikz)
    tikz = re.sub(r"\\foreach \\point in \{([A-Z,]+)\}",
                  lambda m: "\\foreach \\point in {" + ",".join(RENAME_TIKZ[c] for c in m.group(1).split(",")) + "}",
                  tikz)
    return tikz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    problems = {p["problem_id"]: p for p in read_jsonl(args.problems)}
    rows = []
    for pid, by_type in V.items():
        if pid not in problems:
            continue
        p = problems[pid]
        tikz = p["problem"][p["problem"].index(TIKZ_START):] if TIKZ_START in p["problem"] else ""
        for ptype, text in by_type.items():
            variant = text.rstrip() if not tikz else text + "\n" + (rename_tikz(tikz) if ptype == "rename" else tikz)
            rows.append({"problem_id": pid, "type": ptype, "attempt": 0, "variant": variant,
                         "answer": p["answer"], "writer": "claude", "missing_numbers": [],
                         "writer_tokens": None, "original": p["problem"]})
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "stage2_variants.jsonl", rows)
    print(f"{len(rows)} variants for {len({r['problem_id'] for r in rows})} problems")


if __name__ == "__main__":
    main()
