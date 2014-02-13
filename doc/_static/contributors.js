(function($) {

    "use strict";

    // Transparency rendering directives how to map
    // flattened committer data on the client-side template
    var directives = {

        committer: {
            href: function(elem) {
                if(this.html_url) {
                    return this.html_url;
                }
            },

            title: function() {
                return this.name || this.login;
            },

            // Generate centered image using CSS backgroun
            style: function() {
                // Limit Gravatar size
                var imageURL = this.avatar_url + "?size=48";
                return "background-image: url(" + imageURL + ")";
            }


        },

    };

    /**
     * Go through all commits from Github a build a list of committers.
     *
     *  http://stackoverflow.com/a/19200303/315168
     */
    function getAuthors(commits) {
        var authors = [];
        var consumedAuthors = {};

        $.each(commits, function() {

            if(this.author) {

                console.log("Got author", this.author);

                // You may have commits which are not associated with
                // Github username

                var id = this.author.login || this.author.email;

                // Make sure we an image
                if(consumedAuthors[id]) {
                    // Make sure this author does not appear twice
                    return;
                }

                consumedAuthors[id] = true;

                authors.push(this.author);
            }
        });

        return authors;
    }

    $(document).ready(function() {
        var committers = $("#committers");
        var url = committers.attr("data-github-commit-api-url");

        // TODO: Caching the results in LocalStorage here
        // would be nice

        if(!url) {
            return;
        }

        $.getJSON(url, function(data) {
            $(".committer-loader").hide();
            var authors = getAuthors(data);
            $('#committers').render(authors, directives);
            $('#committers').show();
        });
    });

})(jQuery);