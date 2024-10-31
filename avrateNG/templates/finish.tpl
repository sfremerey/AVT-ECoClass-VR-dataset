% rebase('templates/skeleton.tpl', title=title)

<h1 class="mt-5">Post-Fragebogen</h1>
<form id="postquestionnaire" action="/postquestionnaire" method="post">

<h2>SSQ</h2>

<p>
Schätzen Sie die auftretenden Symptome bei der Beantwortung des folgenden Fragebogens bitte ehrlich ein!
</p>
<%

questions = [
  "Allgemeines Unbehagen bzw. Unwohlsein",
  "Ermüdung",
  "Kopfschmerzen",
  "Überanstrengte Augen",
  "Schwierigkeiten mit Sehschärfe",
  "Erhöhte Speichelbildung",
  "Schwitzen",
  "Übelkeit/Erbrechen Konzentrationsschwierigkeiten",
  "Druckgefühl im Kopfbereich",
  "Verschwommene Sicht",
  "Schwindelgefühl (bei geöffneten Augen)",
  "Schwindelgefühl (bei geschlossenen Augen)",
  "Gleichgewichtsstörungen",
  "Magenbeschwerden",
  "Aufstoßen"
]
%>
 <table class="table">
    <thead>
      <tr>
        <td scope="col"></td>
        <td scope="col" class="col-sm-1">Gar nicht</td>
        <td scope="col" class="col-sm-1">Leicht</td>
        <td scope="col" class="col-sm-1">Mittelmäßig</td>
        <td scope="col" class="col-sm-1">Stark</td>
      </tr>
    </thead>
    <tbody>



    % for statement in questions:
      <%
        statement_key = "_".join(statement.lower().split()).replace("?", "").replace(".", "").replace("/", "__").replace(",", "_").replace("ä", "ae").replace("ü", "ue").replace("(", "_").replace(")", "_").replace("ö", "oe").replace("ß", "ss")
      %>
      <tr>
        <td scope="row">{{statement}} </td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}0" value="0" required></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}1" value="1"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}2" value="2"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}3" value="3"></td>
      </tr>
    % end


    </table>
  <hr>

<h2>IPQ</h2>

<%

questions = [

{
    "question": "In der computererzeugten Welt hatte ich den Eindruck, dort gewesen zu sein...",
    "scale": ["überhaupt nicht", "sehr stark"]
},
{
    "question": "Ich hatte das Gefühl, daß die virtuelle Umgebung hinter mir weitergeht.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Ich hatte das Gefühl, nur Bilder zu sehen.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Ich hatte nicht das Gefühl, in dem virtuellen Raum zu sein.",
    "scale": ["hatte nicht das Gefühl", "hatte das Gefühl"]
},
{
    "question": "Ich hatte das Gefühl, in dem virtuellen Raum zu handeln statt etwas von außen zu bedienen.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Ich fühlte mich im virtuellen Raum anwesend.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Wie bewusst war Ihnen die reale Welt, während Sie sich durch die virtuelle Welt bewegten (z.B. Geräusche, Raumtemperatur, andere Personen etc.)?",
    "scale": ["extrem bewusst", "mittelmäßig bewusst", "unbewusst"]
},
{
    "question": "Meine reale Umgebung war mir nicht mehr bewusst.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Ich achtete noch auf die reale Umgebung.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Meine Aufmerksamkeit war von der virtuellen Welt völlig in Bann gezogen.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
},
{
    "question": "Wie real erschien Ihnen die virtuelle Umgebung?",
    "scale": ["vollkommen real", "weder noch", "gar nicht real"]
},
{
    "question": "Wie sehr glich Ihr Erleben der virtuellen Umgebung dem Erleben einer realen Umgebung?",
    "scale": ["überhaupt nicht", "etwas", "vollständig"]
},
{
    "question": "Wie real erschien Ihnen die virtuelle Welt?",
    "scale": ["wie eine vorgestellte Welt", "nicht zu unterscheiden von der realen Welt"]
},
{
    "question": "Die virtuelle Welt erschien mir wirklicher als die reale Welt.",
    "scale": ["trifft gar nicht zu", "trifft völlig zu"]
}

]
%>
 <table class="table">
    <thead>
      <tr>
        <td scope="col" class="col-sm-6"></td>
        <td scope="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col" class="col"></td>
        <td scope="col"></td>
      </tr>
    </thead>
    <tbody>



    % for q in questions:
      <%
        statement_key = "_".join(q["question"].lower().split()).replace("?", "").replace(".", "").replace("/", "__").replace(",", "_").replace("ä", "ae").replace("ü", "ue").replace("(", "_").replace(")", "_").replace("ö", "oe").replace("ß", "ss")
      %>
      <tr>
        <td scope="row">{{q["question"]}} </td>
        <td>{{q["scale"][0]}}</td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}0" value="0" required></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}1" value="1"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}2" value="2"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}3" value="3"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}4" value="4"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}5" value="5"></td>
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}6" value="6"></td>
        <td>{{q["scale"][-1]}}</td>
      </tr>
    % end


    </table>

  <div class="col-lg-2">
    <button type="submit" class="btn btn-success">weiter</button>



  </div>
</form>
