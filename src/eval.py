import click
from pathlib import Path
import numpy as np
import openai

from src.hub import Pipeline
from src.utils import methods

from trulens_eval import Tru
from trulens_eval import OpenAI as fOpenAI
from trulens_eval import Feedback
from trulens_eval import TruLlama
from trulens_eval.feedback import Groundedness


# FIXME: This should be a part of pipeline and should return an evalutor for engine
def create_pipeline():
    path_to_config = Path() / "src" / "conf" / "agent.yaml"
    path_to_secrets = Path() / "secrets"
    pipeline = Pipeline.from_conf(conf_path=path_to_config)
    pipeline.connect_client(secrets_directory=path_to_secrets)
    pipeline.prepare_settings()
    pipeline.prepare_embeddings(parser_type="base")
    return pipeline


def get_query_engine(pipeline: Pipeline):
    engine = pipeline.spawn_query_engine(index_identifier="mostostal-index")
    return engine


def set_ans_relevance_metric(provider: fOpenAI):
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()
    return f_qa_relevance


def set_context_relevance(provider: fOpenAI, context_selection: str):
    f_qs_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )
    return f_qs_relevance


def set_groundness(grounded: Groundedness, context_selection: str):
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(context_selection)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    return f_groundedness


@click.command()
@click.option('--mode', type=click.Choice(['eval', 'dashboard']), default='eval', help='Select mode: eval or dashboard.')
@click.option('--reset-database', is_flag=True, help='Reset the database before processing.')
def main(mode, reset_database):
    tru = Tru()

    if reset_database:
        tru.reset_database()

    if mode == 'eval':
        openai_key = methods.extract_openai_secrets()
        openai.api_key = openai_key["openai_api_key"]

        provider = fOpenAI()
        grounded = Groundedness(groundedness_provider=provider)
        context_selection = TruLlama.select_source_nodes().node.text

        f_qs_relence = set_context_relevance(
            provider=provider, context_selection=context_selection
        )
        f_groundness = set_groundness(
            grounded=grounded, context_selection=context_selection
        )
        f_qa_relevance = set_ans_relevance_metric(provider=provider)

        pipeline = create_pipeline()
        engine = get_query_engine(pipeline=pipeline)


        true_recorder = TruLlama(
            engine,
            app_id="mostostal_main_test",
            feedbacks=[
                f_qa_relevance,
                f_qs_relence,
                f_groundness,
            ],
        )
        eval_questions = [
            "Wyszukaj i wypisz informacje o przedmiocie umowy, zakresie prac. Co powinno być zrealizowane w ramach umowy przez wykonawcę.",
            "Jaki jest termin realizacji przedmiotu umowy, jaki jest harmonogram?",
            "Jaka jest wysokość należnego wykonawcy wynagrodzenia za dostarczenie przedmiotu umowy?",
            "Jakie prawo (prawo z jakiego państwa) jest stosowane w sytuacjach wystąpienia sporów pomiędzy stronami umowy?",
            "Jaki sąd będzie rozstrzygał spory pomiędzy stronami. Czy będzie to: - powszechny (sąd właściwy miejscowo dla siedziby Zamawiającego) - Sąd arbitrażowy : jeśli tak to wyodrębnić",
            "Jakie jest miejsce arbitrażu?",
            "Jaki jest język arbitrażu?",
            "Jaka jest liczba arbitrów?",
            "Czy szczegółowy przedmiot robót wykonanych przez podwykonawcę lub wykonawcę podlega zgłoszeniu przez spółkę na podstawie kodeksu cywilnego (lub analogicznego miejscowego)? Komu tego typu roboty należy zgłosić?",
            "Czy projekt umowy zawiera kaluzule waloryzacyjne?",
            "Czy są zapisy dot. ograniczenia odpowiedzialności odszkodowawczej umownej? Czy odnoszą się do wartości - jeśli tak to do jakiej (wartość lub %).",
            "Czy jest zdefiniowany zakres ograniczenia. Jeśli tak to przedstawić ten zakres.",
            "Czy całkowita odpowiedzialność Spółki (Mostostal Zabrze lub wykonawca) w stosunku do Klienta ograniczona jest do 100% wartości Kontraktu.",
            "Z jakich tytułów Kontrakt przewiduje kary i od jakich wartości kary są liczone?",
            "Czy występują terminy pośrednie podlegające karom umownym? Odpowiedz Tak lub Nie.",
            "Czy kary umowne mogą być należne za terminy pośrednie? Odpowiedz Tak lub Nie.",
            "Czy kary umowne się kumulują? Odpowiedz Tak lub Nie.",
            "Czy występują limity kar? Odpowiedz Tak lub Nie.",
            "Jaki jest limit całkowity dla wszystkich rodzajów kar? Podaj wartość nominalną lub procentową.",
            "Czy kary umowne są podzielone na zakresy? Odpowiedź Tak lub Nie. Jeśli kary są podzielone na zakresy to opisz zakresy i adekwatne wartości kar nominalnie lub procentowo.",
            "Czy zamawiający może jednostronnie ograniczyć zakres robót? Odpowiedz tak lub nie.",
            "Czy występuje łączne ograniczenie wartościowe, do którego możliwe jest ograniczenie zakresu robót? Odpowiedz Tak lub Nie. Jeśli istnieje łączne ograniczenie wartościowe to wskaż kwotę lub wartość procentową.",
            "Czy został podany termin do którego roboty mogą być ograniczane? Odpowiedz Tak lub Nie. Jeśli został podany termin, do którego roboty mogą być ograniczane to wskaż ten termin.",
            "Czy umowa przewiduje zawieszenie lub wstrzymanie robót? Odpowiedz Tak lub Nie. Jeśli umowa przewiduje zawieszenie lub wstrzymanie robót to wskaż ten zapis.",
            "Czy łączny okres zawieszenia/wstrzymania robót jest ograniczony terminem?",
            "Czy zamawiający pokrywa koszty zawieszenia/wstrzymania robót?",
            "Jaki jest sposób ustalenia wynagrodzenia za roboty dodatkowe/zamienne. Opisz lub wskaż, że nie został uregulowany.",
            "Czy roboty dodatkowe lub zamienne są rozliczane jak roboty podstawowe, czy po odbiorze końcowym? Jeśli odpowiedzi nie ma w tekście to wskaż, że status jest nieregulowany.",
            "Czy roboty dodatkowe/zamienne wymagają aneksu czy wystarczy akceptacja przedstawiciela zamawiającego na budowie?",
            "Czy robota dodatkowa (która może być nazwana zamienną) musi być wykonana pomimo braku ustalenia wynagrodzenia? Odpowiedzieć Tak lub Nie bądź wskazać, że nie jest to uregulowane.",
            "Czy robota dodatkowa (która może być nazwana zamienną) musi być wykonana pomimo braku ustalenia zmiany harmonogramu z zamawiającym? Odpowiedzieć Tak lub Nie bądź wskazać, że nie jest to uregulowane.",
            "Czy wskazana jest metoda rozliczenia prac dodatkowych (mogą być nazwane zamiennymi)? Odpowiedzieć tak lub nie. Jeśli jest wskazana wówczas opisać na czym polega metoda rozliczania prac dodatkowych.",
            "Czy procedura rozliczenia prac dodatkowych i zamiennych prowadzi do bieżących rozliczeń. Odpowiedzieć czy są to bieżące rozliczenia, po zakończeniu kontraktu, inne, czy nie jest to uregulowane.",
            "Czy zmiany w zakresie prac (dodanie prac dodatkowych lub zamiennych) wymagają aneksu czy wystarczy akceptacja przedstawiciela Klienta na budowie bądź czy problem jest nieuregulowany przez umowę?",
            "Czy umowa reguluje zasady postępowania w przypadku wzrostu zakresu rzeczowego umowy w trakcie jej realizacji? Odpowiedzieć Tak lub Nie bądź wskazać, że nie jest to uregulowane.",
            "Opisać sposób w jaki umowa definiuje wzrost zakresu rzeczowego umowy.",
            "Czy umowa określa zasady postępowania w związku ze zwiększeniem zakresu rzeczowego umowy? Odpowiedzieć tak lub nie i wskazać jakie są to zasady jeśli zostały określone?",
            "Czy zamawiający ma prawo odstąpienia od umowy bez podania przyczyn? Odpowiedzieć tak lub nie i wskazać sposób rozliczenia wynagrodzenia w przypadku odstąpienia od umowy przez zamawiającego bez podania przyczyn.",
            "Czy dokonanie odbioru robót jest zamknięte pewnym terminem? Odpowiedzieć tak lub nie.",
            "Czy zastrzeżono wyłącznie odbiór „bezusterkowy”? Odpowiedzieć tak lub nie.",
            "Czy zastrzeżono odbiór z możliwością nieistotnych usterek?",
            "Czy odbiór zależy od decyzji inwestora? Odpowiedzieć tak lub nie lub że nie zostało to zdefiniowane. Jeśli jest to zdefiniowane to wskazać od czyjej decyzji zależy odbiór.",
            "Czy zamawiający może używać przedmiotu robót przed odbiorem końcowym? Odpowiedzieć tak lub nie. Jeśli zdefiniowane są warunki to je opisać.",
            "Czy przed odbiorem końcowym ma być wydana dokumentacja sporządzona przez wykonawcę? Odpowiedzieć Tak lub nie lub, że nie jest zdefiniowane. Jeśli dokumentacja powinna być wydana to opisać jaka to powinna być dokumentacja [pozwolenia, dokumentacja techniczna, dokumentacja z badań, inna dokumentacja stanowiąca warunek używania przedmiotu umowy].",
            "Czy procedura odbiorowa jest zdefiniowana?",
            "Czy odbiorów etapów prac można dokonać z listą drobnych wad?",
            "Czy jakikolwiek etap odbioru uzależniony jest od strony trzeciej, na przykład od Inwestora?",
            "Czy terminy odbiorów zastrzeżone w warunkach kontraktowych to terminy zamknięte datami (np. odbiór w ciągu 7 dni od daty zgłoszenia przez Wykonawcę)? Na każde pytanie odpowiedzieć Tak lub nie lub że nie zostało określone.",
            "Czy Wykonawca upoważniony jest do zawieszenia lub odstąpienia od kontraktu w wypadku opóźnienia płatności i/lub późnego wydania akceptacji płatności? Odpowiedzieć tak lub nie.",
            "Czy okres odpowiedzialności za wady jest zamknięty pewnym terminem? Odpowiedzieć Tak lub Nie. Jeśli tak to wskazać termin.",
            "Czy harmonogram prac określa wzajemne powiązania pomiędzy pracami/dostawami Klienta a pracami/dostawami wykonawcy? Odpowiedzieć Tak lub Nie - jeśli tak to wskazać treść zapisów.",
            "Czy została określona procedura zgłaszania zmian/roszczeń przez wykonawcę? Odpowiedzieć Tak lub Nie - jeśli tak to wskazać treść zapisów.",
            "Czy przyjęto zasadę pisemnego zgłaszania wniosków o zmianę/zgłaszania roszczeń? Odpowiedzieć Tak lub nie.",
            "Czy został określony termin na zgłaszanie zmian/roszczeń przez wykonawcę? Odpowiedzieć Tak lub nie. Jeśli tak to wskazać termin.",
            "Czy występuje zrzeczenie się roszczenia/utrata prawa do żądania zmiany w razie jego niezgłoszenia w terminie określonym w procedurze umownej? Odpowiedzieć Tak lub Nie. Jeśli tak to wskazać termin.",
            "Czy procedura umowna wprowadza dla Klienta termin na odniesienie się do wniosku o zmianę/zgłoszenia roszczenia? Odpowiedzieć Tak lub nie. Jeśli tak to wskazać termin.",
            "Czy wykonawca musi wprowadzić / zrealizować zmianę przed uzgodnieniem zmiany wynagrodzenia/harmonogramu z Klientem? W szczególności, czy umowa przewiduje obowiązek ograniczenia opóźnienia i ograniczenia strat z przyczyn leżących po stronie klienta bez określenia procedury ustalania terminu i wynagrodzenia za takie działania nadzwyczajne? Odpowiedzieć Tak lub Nie. Jeśli tak to wskazać zapisy umowy.",
            "Czy umowa przewiduje odpowiedzialność wykonawcy za przyczyny ewentualnych opóźnień na które wykonawca nie ma wpływu? Odpowiedzieć Tak lub nie. Jeśli tak to wskazać zapisy umowy.",
            "Czy umowa przewiduje tak zwane umowne terminy zawite zgłaszania roszczeń (Czy występuje zrzeczenie się roszczeń w razie niewywiązania się z terminów dokonania zgłoszenia przez wykonawcę)? Termin zawity to szczególny rodzaj terminu stanowczego, charakteryzujący się dużym rygorem prawnym, przejawiającym się w tym, że niepodjęcie określonej czynności przez uprawniony podmiot w okresie zakreślonym tym terminem, powoduje definitywne wygaśnięcie przysługującego podmiotowi prawa do tej czynności.",
            "Jakie są terminy zgłaszania wad na poszczególne kategorie robót np. dach, strop, budynek, urządzenia lub inne."
        ]

        for question in eval_questions:
            with true_recorder as recording:
                engine.query(question)

    elif mode == 'dashboard':
        tru.run_dashboard()

if __name__ == "__main__":
    main()
