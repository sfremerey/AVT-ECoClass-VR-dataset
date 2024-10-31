% rebase('templates/skeleton.tpl', title=title)


<h1 class="mt-5">Pre-Fragebogen</h1>

<p class="lead">Bitte beantworten Sie die folgenden Fragen.</p>
<form id="questionnaire" action="/questionnaire" method="post">

<h2>Simulatorkrankheit</h2>

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

<h2>Lärmempfindlichkeit</h2>
<p>
Bitte beantworten Sie die folgenden Fragen ehrlich.
</p>


<%
questions = [
  "Es würde mir nichts ausmachen, an einer lauten Straße zu wohnen, wenn meine Wohnung schön wäre.",
  "Mir fallt Lärm heutzutage mehr auf als früher.",
  "Es sollte niemanden groß stören, wenn ein anderer ab und zu seine Stereoanlage voll aufdreht.",
  "Im Kino stört mich Flüstern und Rascheln von Bonbonpapier.",
  "Ich werde leicht durch Lärm geweckt.",
  "Wenn es an meinem Arbeitsplatz Iaut ist, dann versuche ich, Tür oder Fenster zu schließen oder anderswo weiterzuarbeiten.",
  "Es ärgert mich, wenn meine Nachbarn laut werden.",
  "An die meisten Geräusche gewöhne ich mich ohne große Schwierigkeiten.",
  "Es würde mir etwas ausmachen, wenn eine Wohnung, die ich gerne mieten würde, gegenüber der Feuerwache läge.",
  "Manchmal gehen mir Geräusche auf die Nerven und ärgern mich.",
  "Sogar Musik, die ich eigentlich mag, stört mich, wenn ich mich konzentrieren möchte.",
  "Es würde mich nicht stören, die Alltagsgeräusche meiner Nachbarn (z.B. Schritte, Wasserrauschen) zu hören.",
  "Wenn ich allein sein möchte, stören mich Geräusche von außerhalb.",
  "Ich kann mich gut konzentrieren, egal was um mich herum geschieht.",
  "In der Bibliothek macht es mir nichts aus, wenn sich Leute unterhalten, solange dies leise geschieht.",
  "Oft wünsche ich mir völlige Stille.",
  "Motorräder sollten besser schallgedämpft sein.",
  "Es fällt mir schwer, mich an einem lauten Ort zu entspannen.",
  "Ich werde wütend auf Leute, die Lärm machen, der mich vom Einschlafen oder vom Fortkommen in der Arbeit abhält.",
  "Es würde mir nichts ausmachen, in einer Wohnung mit dünnen Wänden zu leben.",
  "Ich bin geräuschempfindlich."
]
%>

 <table class="table">
    <thead>
      <tr>
        <td scope="col"></td>
        <td scope="col" class="col-sm-1">stimme gar nicht zu </td>
        <td scope="col" class="col-sm-1">stimme nicht zu</td>
        <td scope="col" class="col-sm-1">stimme weder zu noch lehne ich ab</td>
        <td scope="col" class="col-sm-1">stimme zu</td>
        <td scope="col" class="col-sm-1">stimme sehr zu</td>
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
        <td><input class="form-check-input" type="radio" name="radio_{{statement_key}}" id="radio_{{statement_key}}4" value="4"></td>
      </tr>
    % end


    </table>

<p>
Sie werden nun zunächst 4 Trainingssequenzen anschauen.
Bitte weisen Sie mittels Controller-Interaktion die korrekten Geschichten zu genau den Personen zu, die diese Geschichte erzählen.
</p>

  <div class="input-group">
    <input type="hidden" id="screen_size" name="screen_size" placeholder="screen_size" />
    <input type="hidden" id="browser_agent" name="browser_agent" placeholder="browser_agent" />
  </div>

  <div class="col-lg-2">
    <button type="submit" class="btn btn-success">weiter</button>

    % if dev:
      <button type="submit" class="btn btn-success" formnovalidate>skip (for dev)</button>
    % end

  </div>
</form>

