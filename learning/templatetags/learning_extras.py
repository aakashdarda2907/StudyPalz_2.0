# learning/templatetags/learning_extras.py

from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    """Allows dictionary lookup in a Django template."""
    return dictionary.get(key)

@register.filter(name='replace_underscore')
def replace_underscore(value):
    """Replaces underscores with spaces in a string."""
    return value.replace('_', ' ')

# In learning/templatetags/learning_extras.py

from django import template
import markdown2

register = template.Library()

@register.filter(name='markdown_to_html')
def markdown_to_html(markdown_text):
    """
    Converts a string of markdown text to HTML.
    """
    if markdown_text:
        return markdown2.markdown(markdown_text, extras=['fenced-code-blocks', 'tables'])
    return ""