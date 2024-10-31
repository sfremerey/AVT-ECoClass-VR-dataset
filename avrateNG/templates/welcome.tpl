% rebase('templates/skeleton.tpl', title=title)

<h1 class="mt-5">Grundlegender Ablauf des Tests</h1>

<p>
Nach einem kurzen Sehtest füllen Sie zunächst einen Fragebogen aus. Hiernach findet eine Anpassung des Head-Mounted Displays und des Kopfhörers auf die jeweilige Kopfform statt.
Der Linsenabstand kann dabei ebenfalls eingestellt werden.
</p>

<p>
Im Laufe des Tests werden Sie zunächst 4 Trainingssequenzen auf einem Head Mounted Display anschauen.
Während der Videsequenzen werden Sie verschiedene Sprecher:innen aus verschiedenen Richtungen hören, die verschiedene Geschichten über bestimmte Themen erzählen.
Ihre Aufgabe ist es, den verschiedenen Sprechern zuzuhören und so schnell wie möglich herauszufinden, welche Person welche Geschichte erzählt.
Hierbei können Sie Ihren Kopf und Stuhl in jede Richtung drehen, die Sie wünschen.
Die Anzahl und Anordnung der gleichzeitig redenden Sprecher ist zufällig.
Die insgesamt 10 verschiedenen Geschichten lassen sich den folgenden Symbolen zuordnen:
</p>

<img src="/static/stories.png"
width="1307" 
height="376"></img>

<p>
Ihre Aufgabe besteht darin, mittels Controller-Interaktion die korrekten Geschichten zu genau den Personen, die diese Geschichte erzählen, zuzuweisen.
Der Controller sieht so aus:
</p>

<img src="/static/vive_controllers.webp"></img>

<p>
Zur Auswahl einer Geschichte wählen Sie auf dem Trackpad des Controllers (2) zuerst mittels kreisförmigen Bewegungen die Geschichte aus, von der Sie denken, dass sie von einer Person erzählt wird.
Dann zielen Sie mit dem roten Controller-Ray auf den entsprechenden "OK" Button über der jeweiligen Person und weisen mit dem Drücken des Trigger Buttons (7) die Geschichte zu.
Nach der Zuweisung verschwindet die Geschichte aus dem Auswahlrad.
Sie müssen dabei sicherstellen, dass die entsprechende Geschichte nach der Zuweisung im entsprechenden Feld sichtbar ist, dies sieht beispielsweise so aus:
</p>

<img src="/static/input_field.png"
width="441" 
height="514"></img>

<p>
Falls Sie denken, dass eine Ihrer zugewiesenen Geschichten falsch war, können Sie jederzeit durch Drücken des roten "X" Buttons mittels des Trigger Buttons (7) Ihre Zuweisung rückgängig machen.
Sie sollen die Aufgabe so schnell wie möglich erfüllen, d.h. allen sprechenden Personen schnellstmöglich die passende Geschichte zuweisen.
Nicht sprechenden Personen darf keine Geschichte zugewiesen werden, die Felder über diesen Personen bleiben leer.
Für die Erfüllung der Aufgabe haben Sie pro Videosequenz 2 Minuten Zeit.
Wenn Sie mit der Aufgabe fertig sind, drücken Sie den Menu Button (1).
Hiernach wird Ihre Auswahl automatisch gespeichert und die Trainingssequenz wird beendet.
Falls Sie nach 2 Minuten die Erfüllung der Aufgaben nicht abgeschlossen haben, wird Ihre aktuelle Auswahl automatisch gespeichert und die Traningssequenz wird beendet.
</p>


<p>
Nach 4 Trainingssequenzen werden Sie weitere 9 Videosequenzen sehen.
</p>

<p>
Nach jeder betrachteten Videosequenz nehmen Sie bitte das Head-Mounted Display ab und füllen am Bildschirm einen Fragebogen aus.
Hierbei ist es wichtig, dass Sie den Fragebogen immer ehrlich beantworten.
Sie werden genug Zeit haben, den entsprechenden Fragebogen auszufüllen.
Nach der Beantwortung des Fragebogens werden Sie ebenfalls Zeit für eine Pause haben.
</p>

<p>
Wenn Ihnen bei der Betrachtung des Materials unwohl wird oder Sie sich nicht in der Lage fühlen, den Test weiter fortzuführen, geben Sie bitte dem Versuchsleiter Bescheid.
</p>


<!-- if questionnaire is deactivated the system will skip it -->
<a class="btn btn-large btn-success" href="/questionnaire"  id="start">Start</a>
</p>

